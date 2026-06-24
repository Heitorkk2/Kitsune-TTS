import os
import json
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from kitsune.data.dataset import KitsuneDataset
from kitsune.data.collate import KitsuneCollate
from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.model.discriminator import MultiPeriodDiscriminator
from kitsune.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from kitsune.utils import save_checkpoint, load_checkpoint, update_train_state, get_logger
from kitsune.data.symbols import symbols
from kitsune.data.audio import mel_spectrogram_torch

class KitsuneTrainer:
    def __init__(self, config_path: str, model_dir: str):
        self.config_path = config_path
        self.model_dir = model_dir
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.hparams = json.load(f)
            
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger = get_logger("KitsuneTrainer", os.path.join(self.model_dir, "train.log"))
        
        # Save configs to output folder for reproducibility
        self._save_configs()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed Precision Setup
        self.precision = self.hparams["train"].get("precision", "fp32")
        self.amp_enabled = self.precision in ["bf16", "fp16"] and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else (torch.float16 if self.precision == "fp16" else torch.float32)
        # GradScaler is only meaningful for fp16; bf16 has enough dynamic range
        # that loss scaling is unnecessary (and enabling it can distort grads).
        self.scaler = GradScaler(enabled=(self.precision == "fp16" and torch.cuda.is_available()))
        self.grad_clip = self.hparams["train"].get("grad_clip", None)
        self.embedding_only = (
            self.hparams["train"].get("speaker_finetune_mode") == "embedding_only"
        )
        
        # WandB Setup
        self.use_wandb = self.hparams["train"].get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=self.hparams["train"].get("wandb_project", "kitsune-tts"), config=self.hparams)
            except ImportError:
                self.logger.warning("wandb is not installed. Run `pip install wandb` to use it.")
                self.use_wandb = False
                
        self._build_models()
    
    def _save_configs(self):
        """Saves model_config.json and train_config.json separately in the output folder."""
        # Model config = architecture only
        model_config_path = os.path.join(self.model_dir, "model_config.json")
        with open(model_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.hparams["model"],
                "data": self.hparams["data"],
                "speakers": self.hparams.get("speakers", {})
            }, f, indent=2)
        
        # Train config = hyperparameters only
        train_config_path = os.path.join(self.model_dir, "train_config.json")
        with open(train_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "train": self.hparams["train"]
            }, f, indent=2)
        
    def _build_models(self):
        # Resolve number of speakers dynamically from the "speakers" map
        speakers_map = self.hparams.get("speakers", {})
        if speakers_map:
            n_speakers = len(speakers_map)
        else:
            n_speakers = self.hparams["model"].get("n_speakers", 0)
        
        vocab_size = len(symbols)
        
        # Override n_speakers in model params with the resolved count
        model_params = dict(self.hparams["model"])
        model_params["n_speakers"] = n_speakers
        
        self.net_g = SynthesizerTrn(
            vocab_size,
            self.hparams["data"]["filter_length"] // 2 + 1,
            self.hparams["train"]["segment_size"] // self.hparams["data"]["hop_length"],
            **model_params
        ).to(self.device)
        
        self.net_d = None
        if not self.embedding_only:
            self.net_d = MultiPeriodDiscriminator(use_spectral_norm=False).to(self.device)
        
        if hasattr(torch, "compile") and self.hparams["train"].get("use_compile", False):
            self.logger.info("Compiling models with torch.compile() for extreme speed...")
            # dynamic=False is faster, but if input lengths vary widely it might recompile often
            self.net_g = torch.compile(self.net_g, dynamic=True)
            if self.net_d is not None:
                self.net_d = torch.compile(self.net_d, dynamic=True)
        
        self.optim_g = torch.optim.AdamW(self.net_g.parameters(), self.hparams["train"]["learning_rate"], betas=self.hparams["train"]["betas"], eps=self.hparams["train"]["eps"])
        self.optim_d = (
            torch.optim.AdamW(self.net_d.parameters(), self.hparams["train"]["learning_rate"], betas=self.hparams["train"]["betas"], eps=self.hparams["train"]["eps"])
            if self.net_d is not None
            else None
        )

        self.global_step = 0
        self.start_epoch = 1

    def _build_schedulers(self):
        """Exponential LR decay per epoch (VITS-style), using train.lr_decay."""
        lr_decay = self.hparams["train"].get("lr_decay", 1.0)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=lr_decay)
        # Fast-forward the decay for the epochs already completed when resuming.
        # (Stepping avoids the last_epoch>=0 'initial_lr' requirement on a fresh optimizer.)
        for _ in range(self.start_epoch - 1):
            self.scheduler_g.step()
            self.scheduler_d.step()

    def _clip_grads(self, optimizer, module):
        """Unscale (a no-op when the scaler is disabled) then clip grad norm. Returns True if grads are valid, False if NaN."""
        if self.grad_clip is None:
            for p in module.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return False
            return True
            
        self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
        return torch.isfinite(norm)

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _prepare_eval_inputs(self):
        """
        Phonemize the eval sentence once and cache phoneme-id tensors.
        Returns a list of (speaker_name, sid_tensor, ids_tensor) or None on failure.
        """
        from kitsune.phonemizer.phonemizer import EspeakPhonemizer
        from kitsune.data.symbols import cleaned_text_to_sequence

        text = self.hparams["train"].get(
            "eval_text",
            "O rápido cão marrom pulou sobre a raposa preguiçosa, será que ela acordou?"
        )
        lang = self.hparams["train"].get("eval_lang", "pt-br")

        phonemizer = EspeakPhonemizer()
        ids = cleaned_text_to_sequence(phonemizer.phonemize(text, lang))
        if not ids:
            raise RuntimeError("Eval sentence phonemized to an empty sequence.")
        ids_tensor = torch.LongTensor(ids)

        speakers = self.hparams.get("speakers", {})
        if speakers:
            return [(name, torch.LongTensor([sid]), ids_tensor) for name, sid in speakers.items()]
        return [("default", None, ids_tensor)]

    def _run_eval(self, epoch):
        """
        Generate one short wav per speaker into model_dir/eval/checkpoint-{step}/.
        Cheap by construction: batch=1, one fixed short sentence, no_grad, and it
        bypasses the torch.compile wrapper so no inference graph is ever compiled.
        Any failure is logged and swallowed -- eval must never kill training.
        """
        try:
            from scipy.io.wavfile import write as write_wav
            import numpy as np

            if not hasattr(self, "_eval_inputs"):
                try:
                    self._eval_inputs = self._prepare_eval_inputs()
                except Exception as e:
                    # Preparation is deterministic (phonemizer/config); if it fails
                    # once it will fail every time, so disable instead of paying
                    # the espeak init cost at every interval.
                    self._eval_inputs = None
                    self.logger.warning(f"Eval disabled (setup failed): {e}")
            if self._eval_inputs is None:
                return

            # Eager module (skip the compile wrapper: infer would trigger a slow recompile)
            net_g = getattr(self.net_g, "_orig_mod", self.net_g)

            out_dir = os.path.join(self.model_dir, "eval", f"checkpoint-{self.global_step}")
            os.makedirs(out_dir, exist_ok=True)
            sr = self.hparams["data"]["sampling_rate"]

            was_training = net_g.training
            net_g.eval()
            try:
                with torch.no_grad():
                    for name, sid, ids in self._eval_inputs:
                        x = ids.unsqueeze(0).to(self.device)
                        x_lengths = torch.LongTensor([ids.size(0)]).to(self.device)
                        sid_dev = sid.to(self.device) if sid is not None else None
                        audio, *_ = net_g.infer(x, x_lengths, sid=sid_dev,
                                                noise_scale=0.667, length_scale=1.0)
                        wav = audio[0, 0].float().cpu().numpy()
                        # An untrained/early model can emit NaN (e.g. fresh SDP);
                        # nan_to_num keeps the wav readable instead of corrupt.
                        wav = np.clip(np.nan_to_num(wav), -1.0, 1.0)
                        write_wav(os.path.join(out_dir, f"{name}.wav"), sr,
                                  (wav * 32767).astype(np.int16))
                        del audio, wav
            finally:
                if was_training:
                    net_g.train()

            self.logger.info(f"Eval samples written to {out_dir}")
        except Exception as e:
            # Transient generation failure: log and retry at the next interval.
            self.logger.warning(f"Eval generation failed (training continues): {e}")

    def _expand_speakers(self, num_new_speakers: int):
        """Dynamically expands the embedding layer to accommodate new speakers during fine-tuning."""
        if not hasattr(self.net_g, 'emb_g'):
            return
            
        old_emb = self.net_g.emb_g.weight.data
        old_num = old_emb.size(0)
        new_num = old_num + num_new_speakers
        gin_channels = old_emb.size(1)
        
        self.logger.info(f"Expanding speaker embedding from {old_num} to {new_num} slots.")
        
        new_emb_layer = torch.nn.Embedding(new_num, gin_channels)
        new_emb_layer.weight.data[:old_num] = old_emb
        
        # Initialize new embeddings with the distribution of the base speakers
        with torch.no_grad():
            mean = old_emb.mean(dim=0)
            std = old_emb.std(dim=0)
            new_emb_layer.weight.data[old_num:] = torch.randn(new_num - old_num, gin_channels).to(old_emb.device) * std + mean
            
        self.net_g.emb_g = new_emb_layer.to(self.device)
        
        # Re-init optimizer to track new parameters
        self.optim_g = torch.optim.AdamW(self.net_g.parameters(), self.hparams["train"]["learning_rate"], betas=self.hparams["train"]["betas"], eps=self.hparams["train"]["eps"])
        
    def resume_from_checkpoint(self, checkpoint_path: str = None):
        """
        Resumes training from a checkpoint. If no path is given, looks for checkpoint_latest.pth.
        """
        if checkpoint_path is None:
            latest = os.path.join(self.model_dir, "checkpoint_latest.pth")
            if os.path.isfile(latest):
                checkpoint_path = latest
            else:
                self.logger.info("No checkpoint found, starting from scratch.")
                return
        
        step, epoch = load_checkpoint(checkpoint_path, self.net_g, self.net_d, self.optim_g, self.optim_d)
        self.global_step = step
        self.start_epoch = epoch
        self.logger.info(f"Resumed from step {step}, epoch {epoch}")

    def fine_tune(self, base_checkpoint: str, new_dataset_file: str, new_speakers_count: int = 0):
        """
        Sets up the trainer for fine-tuning on a new voice.
        """
        self._expand_speakers(new_speakers_count)
        
        step, epoch = load_checkpoint(base_checkpoint, self.net_g)
        self.logger.info(f"Loaded base checkpoint from step {step}")
        
        # Override the dataset file for fine-tuning
        self.hparams["data"]["training_files"] = new_dataset_file
        
        # Lower LR for fine-tuning
        for param_group in self.optim_g.param_groups:
            param_group['lr'] = self.hparams["train"]["learning_rate"] * 0.5
        for param_group in self.optim_d.param_groups:
            param_group['lr'] = self.hparams["train"]["learning_rate"] * 0.5
            
        self.train()

    def fine_tune_speaker_embedding(
        self,
        speaker_id: int,
        max_steps: int = 2000,
        embedding_l2: float = 1e-4,
        save_interval: int = 250,
        save_dtype: str = "fp16",
    ):
        """Adapt only one speaker embedding while preserving every old voice.

        All shared generator parameters are frozen and the discriminator is not
        used. The optimizer can update only ``emb_g.weight``; a gradient mask
        plus an exact post-step restore protects every non-target row. This is
        the safe, low-VRAM path intended for short Colab/T4 adaptations.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if save_interval < 1:
            raise ValueError("save_interval must be positive.")
        if save_dtype not in {"fp16", "fp32"}:
            raise ValueError("save_dtype must be 'fp16' or 'fp32'.")

        net_g = getattr(self.net_g, "_orig_mod", self.net_g)
        if not hasattr(net_g, "emb_g"):
            raise ValueError("Speaker embedding fine-tuning requires a multi-speaker model.")
        if not 0 <= speaker_id < net_g.emb_g.num_embeddings:
            raise ValueError(
                f"speaker_id {speaker_id} is outside [0, {net_g.emb_g.num_embeddings - 1}]."
            )

        speaker_names = {
            int(sid): name for name, sid in self.hparams.get("speakers", {}).items()
        }
        speaker_name = speaker_names.get(speaker_id, f"speaker_{speaker_id}")

        train_dataset = KitsuneDataset(
            self.hparams["data"]["training_files"],
            filter_length=self.hparams["data"]["filter_length"],
            hop_length=self.hparams["data"]["hop_length"],
            win_length=self.hparams["data"]["win_length"],
            sampling_rate=self.hparams["data"]["sampling_rate"],
        )
        dataset_speaker_ids = {
            int(row[1]) for row in train_dataset.audiopaths_and_text
        }
        if dataset_speaker_ids != {speaker_id}:
            raise ValueError(
                "The embedding-only dataset must contain exactly the target speaker ID. "
                f"Expected {{{speaker_id}}}, found {sorted(dataset_speaker_ids)}."
            )

        from kitsune.data.sampler import TextBucketSampler

        collate_fn = KitsuneCollate()
        batch_sampler = TextBucketSampler(
            train_dataset,
            int(self.hparams["train"]["batch_size"]),
            shuffle=True,
        )
        num_workers = int(self.hparams["train"].get("num_workers", 2))
        loader_options = {
            "dataset": train_dataset,
            "num_workers": num_workers,
            "batch_sampler": batch_sampler,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers:
            loader_options["prefetch_factor"] = int(
                self.hparams["train"].get("prefetch_factor", 2)
            )
            loader_options["persistent_workers"] = True
        train_loader = DataLoader(**loader_options)
        if not len(train_loader):
            raise ValueError("Fine-tuning dataset is empty.")

        # The discriminator and its optimizer are unnecessary for embedding-only
        # reconstruction and otherwise waste a meaningful amount of T4 VRAM.
        if self.net_d is not None:
            self.net_d.to("cpu")
        self.optim_d = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        for parameter in net_g.parameters():
            parameter.requires_grad_(False)
        embedding = net_g.emb_g.weight
        embedding.requires_grad_(True)
        protected_rows = embedding.detach().clone()
        initial_target = protected_rows[speaker_id].clone()

        gradient_mask = torch.zeros_like(embedding)
        gradient_mask[speaker_id].fill_(1)
        gradient_hook = embedding.register_hook(lambda gradient: gradient * gradient_mask)

        self.optim_g = torch.optim.Adam(
            [embedding],
            lr=float(self.hparams["train"]["learning_rate"]),
            betas=tuple(self.hparams["train"].get("betas", (0.9, 0.99))),
            eps=float(self.hparams["train"].get("eps", 1e-8)),
        )
        grad_clip = float(self.hparams["train"].get("grad_clip", 5.0))
        mel_weight = float(self.hparams["train"].get("mel_weight", 45.0))
        adaptation_step = 0
        base_step = self.global_step
        epoch = 0
        latest_losses = {}

        def restore_protected_rows():
            with torch.no_grad():
                embedding[:speaker_id].copy_(protected_rows[:speaker_id])
                embedding[speaker_id + 1 :].copy_(protected_rows[speaker_id + 1 :])

        def save_outputs(interrupted=False, full_model=True):
            output_dtype = torch.float16 if save_dtype == "fp16" else torch.float32
            delta_path = os.path.join(self.model_dir, "speaker_embedding_delta.pth")
            torch.save(
                {
                    "format": "kitsune-speaker-embedding-delta-v1",
                    "speaker": speaker_name,
                    "speaker_id": speaker_id,
                    "embedding": embedding[speaker_id].detach().cpu().to(output_dtype),
                    "adaptation_step": adaptation_step,
                    "base_step": base_step,
                    "interrupted": interrupted,
                },
                delta_path,
            )

            model_path = os.path.join(self.model_dir, f"latest_model_{save_dtype}.pth")
            if full_model:
                state_dict = {
                    key: value.detach().cpu().to(output_dtype)
                    if value.is_floating_point()
                    else value.detach().cpu()
                    for key, value in net_g.state_dict().items()
                }
                torch.save(
                    {
                        "generator": state_dict,
                        "step": self.global_step,
                        "adaptation_step": adaptation_step,
                        "epoch": epoch,
                        "speaker_finetune": {
                            "mode": "embedding_only",
                            "speaker": speaker_name,
                            "speaker_id": speaker_id,
                            "old_speakers_preserved": True,
                        },
                    },
                    model_path,
                )
            update_train_state(
                self.model_dir,
                self.global_step,
                epoch,
                losses={**latest_losses, "adaptation_step": adaptation_step},
            )
            return (model_path if full_model else None), delta_path

        self.logger.info(
            "Embedding-only fine-tune: speaker=%s (ID %d), trainable=%d, max_steps=%d",
            speaker_name,
            speaker_id,
            embedding[speaker_id].numel(),
            max_steps,
        )
        net_g.eval()
        progress = tqdm(total=max_steps, desc=f"Adapting {speaker_name}")
        interrupted = False
        try:
            while adaptation_step < max_steps:
                epoch += 1
                for batch in train_loader:
                    x = batch[0].to(self.device, non_blocking=True)
                    x_lengths = batch[1].to(self.device, non_blocking=True)
                    spec = batch[2].to(self.device, non_blocking=True)
                    spec_lengths = batch[3].to(self.device, non_blocking=True)
                    y = batch[4].to(self.device, non_blocking=True)
                    sid = batch[6].to(self.device, non_blocking=True)

                    self.optim_g.zero_grad(set_to_none=True)
                    with autocast(
                        device_type=self.device.type,
                        enabled=self.amp_enabled,
                        dtype=self.amp_dtype,
                    ):
                        y_hat, duration_loss, _, ids_slice, _, z_mask, (
                            _, z_p, m_p, logs_p, _, logs_q
                        ) = net_g(x, x_lengths, spec, spec_lengths, sid=sid)

                        from kitsune.model.commons import slice_segments

                        y_slice = slice_segments(
                            y,
                            ids_slice * self.hparams["data"]["hop_length"],
                            self.hparams["train"]["segment_size"],
                        )
                        common_length = min(y_slice.size(2), y_hat.size(2))
                        y_slice = y_slice[:, :, :common_length]
                        y_hat = y_hat[:, :, :common_length]
                        mel = mel_spectrogram_torch(
                            y_slice.squeeze(1),
                            self.hparams["data"]["filter_length"],
                            self.hparams["data"]["n_mel_channels"],
                            self.hparams["data"]["sampling_rate"],
                            self.hparams["data"]["hop_length"],
                            self.hparams["data"]["win_length"],
                            self.hparams["data"]["mel_fmin"],
                            self.hparams["data"]["mel_fmax"],
                        )
                        mel_hat = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            self.hparams["data"]["filter_length"],
                            self.hparams["data"]["n_mel_channels"],
                            self.hparams["data"]["sampling_rate"],
                            self.hparams["data"]["hop_length"],
                            self.hparams["data"]["win_length"],
                            self.hparams["data"]["mel_fmin"],
                            self.hparams["data"]["mel_fmax"],
                        )
                        loss_mel = torch.nn.functional.l1_loss(mel, mel_hat) * mel_weight
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                        loss_duration = duration_loss.mean()
                        loss_embedding = torch.nn.functional.mse_loss(
                            embedding[speaker_id], initial_target
                        ) * embedding_l2
                        loss_total = loss_mel + loss_kl + loss_duration + loss_embedding

                    if not torch.isfinite(loss_total):
                        raise FloatingPointError(
                            f"Non-finite speaker adaptation loss at step {adaptation_step}."
                        )
                    self.scaler.scale(loss_total).backward()
                    self.scaler.unscale_(self.optim_g)
                    torch.nn.utils.clip_grad_norm_([embedding], grad_clip)
                    self.scaler.step(self.optim_g)
                    self.scaler.update()
                    restore_protected_rows()

                    adaptation_step += 1
                    self.global_step += 1
                    latest_losses = {
                        "total": round(float(loss_total.detach().item()), 5),
                        "mel": round(float(loss_mel.detach().item()), 5),
                        "kl": round(float(loss_kl.detach().item()), 5),
                        "duration": round(float(loss_duration.detach().item()), 5),
                    }
                    progress.update(1)
                    progress.set_postfix(total=f"{latest_losses['total']:.3f}")

                    if adaptation_step % save_interval == 0:
                        # Frequent checkpoints are tiny embedding deltas. The
                        # full ~39M model is written only once at the end.
                        save_outputs(full_model=False)
                    if adaptation_step >= max_steps:
                        break
        except KeyboardInterrupt:
            interrupted = True
            self.logger.info("Speaker adaptation interrupted; saving current embedding.")
        finally:
            restore_protected_rows()
            gradient_hook.remove()
            progress.close()

        model_path, delta_path = save_outputs(interrupted=interrupted)
        if not torch.equal(embedding[:speaker_id], protected_rows[:speaker_id]) or not torch.equal(
            embedding[speaker_id + 1 :], protected_rows[speaker_id + 1 :]
        ):
            raise RuntimeError("Protected speaker embeddings changed during adaptation.")
        self.logger.info("Saved adapted model: %s", model_path)
        self.logger.info("Saved portable speaker delta: %s", delta_path)
        return model_path, delta_path

    def _build_validation_loader(self, collate_fn, num_workers):
        """Builds the validation DataLoader from data.validation_files.

        Prior to this function, train() never read "validation_files" -- the key
        existed in config.json but had no consumer, so the validation set was
        never loaded or used as a metric (confirmed bug: no reference to "validation"
        existed in this file prior to this patch). This ports the proven mechanism
        from pod_prosody_update/project/kitsune/trainer.py, which already feeds
        the production dashboard.
        """
        validation_files = self.hparams["data"].get("validation_files")
        if not validation_files:
            self.logger.warning("validation_files não configurado; validação de duração desligada.")
            return None
        if not os.path.isfile(validation_files):
            raise FileNotFoundError(f"validation_files configurado mas arquivo não existe: {validation_files}")

        dataset = KitsuneDataset(
            validation_files,
            filter_length=self.hparams["data"]["filter_length"],
            hop_length=self.hparams["data"]["hop_length"],
            win_length=self.hparams["data"]["win_length"],
            sampling_rate=self.hparams["data"]["sampling_rate"],
        )
        loader_options = {
            "dataset": dataset,
            "batch_size": int(
                self.hparams["train"].get("validation_batch_size", self.hparams["train"]["batch_size"])
            ),
            "shuffle": False,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers:
            loader_options["prefetch_factor"] = int(
                self.hparams["train"].get("prefetch_factor", 2)
            )
            loader_options["persistent_workers"] = True
        return DataLoader(**loader_options)

    def _run_validation(self, validation_loader):
        """Measures duration loss on the validation set using a fixed seed, without updating the model."""
        if validation_loader is None:
            return None

        net_g = getattr(self.net_g, "_orig_mod", self.net_g)
        was_training = net_g.training
        total_loss = 0.0
        total_examples = 0
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        net_g.eval()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(self.hparams["train"].get("eval_seed", self.hparams["train"].get("seed", 42))))
                with torch.no_grad():
                    for batch in validation_loader:
                        x = batch[0].to(self.device, non_blocking=True)
                        x_lengths = batch[1].to(self.device, non_blocking=True)
                        spec = batch[2].to(self.device, non_blocking=True)
                        spec_lengths = batch[3].to(self.device, non_blocking=True)
                        sid = batch[6].to(self.device, non_blocking=True)
                        with autocast(device_type=self.device.type if hasattr(self.device, "type") else ("cuda" if torch.cuda.is_available() else "cpu"),
                                      enabled=self.amp_enabled, dtype=self.amp_dtype):
                            duration_loss = net_g.duration_loss(
                                x, x_lengths, spec, spec_lengths, sid=sid
                            )
                        batch_size = int(x.size(0))
                        total_loss += float(duration_loss.mean().item()) * batch_size
                        total_examples += batch_size
        finally:
            if was_training:
                net_g.train()

        if not total_examples:
            raise RuntimeError("Validation dataset está vazio.")
        result = total_loss / total_examples
        self.logger.info(f"Validation step {self.global_step} - duration_loss: {result:.6f}")
        return result


    def _save(self, epoch, tag=None, losses=None):
        """Unified save: checkpoint + train_state.json update."""
        save_checkpoint(
            self.net_g, self.net_d,
            self.optim_g, self.optim_d,
            self.global_step, epoch,
            self.model_dir, tag=tag
        )
        update_train_state(self.model_dir, self.global_step, epoch, losses=losses)
        
        # Always export a clean inference-only model (Generator weights only, no D, no optimizers)
        if tag == "latest":
            inference_path = os.path.join(self.model_dir, "latest_model.pth")
            torch.save({
                'generator': self.net_g.state_dict(),
                'step': self.global_step,
                'epoch': epoch,
            }, inference_path)

    def train(self):
        if self.embedding_only:
            raise RuntimeError(
                "speaker_finetune_mode=embedding_only must use "
                "fine_tune_speaker_embedding(), not the adversarial train() loop."
            )
        train_dataset = KitsuneDataset(self.hparams["data"]["training_files"], 
                                       filter_length=self.hparams["data"]["filter_length"],
                                       hop_length=self.hparams["data"]["hop_length"],
                                       win_length=self.hparams["data"]["win_length"],
                                       sampling_rate=self.hparams["data"]["sampling_rate"])
        collate_fn = KitsuneCollate()
        from kitsune.data.sampler import TextBucketSampler
        
        batch_size = self.hparams["train"]["batch_size"]
        bucket_sampler = TextBucketSampler(train_dataset, batch_size, shuffle=True)
        
        cpu_count = os.cpu_count() or 1
        optimal_workers = int(
            self.hparams["train"].get(
                "num_workers", min(8, max(1, cpu_count // 2))
            )
        )
        if optimal_workers < 0:
            raise ValueError("train.num_workers must be zero or a positive integer.")
        prefetch_factor = int(self.hparams["train"].get("prefetch_factor", 2))
        
        # Use batch_sampler for proper custom bucketing structure
        loader_options = {
            "dataset": train_dataset,
            "num_workers": optimal_workers,
            "batch_sampler": bucket_sampler,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available(),
        }
        if optimal_workers:
            loader_options["prefetch_factor"] = prefetch_factor
            loader_options["persistent_workers"] = True
        train_loader = DataLoader(**loader_options)
        # Actually loads the validation set (see _build_validation_loader): previously,
        # data.validation_files was never read and training ran
        # without any validation criteria.
        validation_loader = self._build_validation_loader(collate_fn, optimal_workers)
        epochs = self.hparams["train"]["epochs"]
        total_steps = epochs * len(train_loader)

        # Build LR schedulers now (optimizers are final: fine_tune/resume ran first).
        self._build_schedulers()

        self.logger.info(f"Starting training loop... ({epochs} epochs, {total_steps} total steps)")
        self.net_g.train()
        self.net_d.train()
        
        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training Progress")

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                for batch_idx, batch in enumerate(train_loader):
                    # Move batch to device asynchronously
                    x, x_lengths = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)
                    spec, spec_lengths = batch[2].to(self.device, non_blocking=True), batch[3].to(self.device, non_blocking=True)
                    y, y_lengths = batch[4].to(self.device, non_blocking=True), batch[5].to(self.device, non_blocking=True)
                    sid = batch[6].to(self.device, non_blocking=True)

                    self.optim_d.zero_grad(set_to_none=True)

                    # Forward Generator
                    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=self.amp_enabled, dtype=self.amp_dtype):
                        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths, sid=sid)
                        
                        import kitsune.model.commons as commons
                        y_slice = commons.slice_segments(y, ids_slice * self.hparams["data"]["hop_length"], self.hparams["train"]["segment_size"])
                        
                        if y_slice.size(2) < y_hat.size(2): y_hat = y_hat[:, :, :y_slice.size(2)]
                        elif y_slice.size(2) > y_hat.size(2): y_slice = y_slice[:, :, :y_hat.size(2)]
                        
                        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y_slice, y_hat.detach())
                        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

                    if torch.isfinite(loss_disc):
                        self.scaler.scale(loss_disc).backward()
                        if self._clip_grads(self.optim_d, self.net_d):
                            self.scaler.step(self.optim_d)
                        else:
                            self.logger.warning(f"NaN/Inf in Discriminator gradients at step {self.global_step}. Skipping D step.")
                    else:
                        self.logger.warning(f"NaN/Inf in Discriminator loss at step {self.global_step}. Skipping D step.")

                    # Generator Loss. Freeze D weights while retaining the
                    # gradient from D(y_hat) back into the generator.
                    self.optim_g.zero_grad(set_to_none=True)
                    self.optim_d.zero_grad(set_to_none=True)
                    self._set_requires_grad(self.net_d, False)
                    try:
                        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=self.amp_enabled, dtype=self.amp_dtype):
                            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(
                                y_slice, y_hat, real_requires_grad=False
                            )

                            mel = mel_spectrogram_torch(y_slice.squeeze(1), self.hparams["data"]["filter_length"], self.hparams["data"]["n_mel_channels"], self.hparams["data"]["sampling_rate"], self.hparams["data"]["hop_length"], self.hparams["data"]["win_length"], self.hparams["data"]["mel_fmin"], self.hparams["data"]["mel_fmax"])
                            mel_hat = mel_spectrogram_torch(y_hat.squeeze(1), self.hparams["data"]["filter_length"], self.hparams["data"]["n_mel_channels"], self.hparams["data"]["sampling_rate"], self.hparams["data"]["hop_length"], self.hparams["data"]["win_length"], self.hparams["data"]["mel_fmin"], self.hparams["data"]["mel_fmax"])

                            loss_mel = torch.nn.functional.l1_loss(mel, mel_hat) * 45
                            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                            loss_fm = feature_loss(fmap_r, fmap_g)
                            loss_gen, _ = generator_loss(y_d_hat_g)
                            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + l_length.mean()

                        if torch.isfinite(loss_gen_all):
                            self.scaler.scale(loss_gen_all).backward()
                            if self._clip_grads(self.optim_g, self.net_g):
                                self.scaler.step(self.optim_g)
                            else:
                                self.logger.warning(f"NaN/Inf in Generator gradients at step {self.global_step}. Skipping G step.")
                            self.scaler.update()
                        else:
                            self.logger.warning(f"NaN/Inf in Generator loss at step {self.global_step}. Skipping G step.")
                    finally:
                        self._set_requires_grad(self.net_d, True)

                    self.global_step += 1
                    
                    pbar.update(1)
                    
                    if self.global_step % self.hparams["train"]["log_interval"] == 0:
                        g_loss_val = loss_gen_all.item()
                        d_loss_val = loss_disc.item()
                        pbar.set_postfix({"Epoch": epoch, "G_Loss": f"{g_loss_val:.3f}", "D_Loss": f"{d_loss_val:.3f}"})
                        self.logger.info(f"Epoch: {epoch}, Step: {self.global_step} - G_Loss: {g_loss_val:.4f}, D_Loss: {d_loss_val:.4f}")
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "loss/g_total": g_loss_val,
                                "loss/d_total": d_loss_val,
                                "loss/mel": loss_mel.item(),
                                "loss/kl": loss_kl.item(),
                                "loss/feature_matching": loss_fm.item(),
                                "loss/duration": l_length.mean().item(),
                                "epoch": epoch,
                                "global_step": self.global_step
                            })

                    if self.global_step % self.hparams["train"].get("eval_interval", 1000) == 0:
                        duration_val = self._run_validation(validation_loader)
                        if duration_val is not None:

                            if self.use_wandb:
                                import wandb
                                wandb.log({
                                    "val/duration": duration_val,
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                })
                        self._run_eval(epoch)

                    if self.global_step % self.hparams["train"]["save_interval"] == 0:
                        current_losses = {
                            "g_total": round(loss_gen_all.item(), 4),
                            "d_total": round(loss_disc.item(), 4),
                            "mel": round(loss_mel.item(), 4),
                            "kl": round(loss_kl.item(), 4),
                            "feature_matching": round(loss_fm.item(), 4),
                            "duration": round(l_length.mean().item(), 4),
                        }
                        self._save(epoch, losses=current_losses)

                # Save latest at the end of each epoch
                # We extract the last batch's losses for the epoch summary
                try:
                    current_losses = {
                        "g_total": round(loss_gen_all.item(), 4),
                        "d_total": round(loss_disc.item(), 4),
                        "mel": round(loss_mel.item(), 4),
                        "kl": round(loss_kl.item(), 4),
                        "feature_matching": round(loss_fm.item(), 4),
                        "duration": round(l_length.mean().item(), 4),
                    }
                except:
                    current_losses = {}
                self._save(epoch, tag="latest", losses=current_losses)

                # Decay the learning rate once per epoch (VITS-style).
                self.scheduler_g.step()
                self.scheduler_d.step()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted. Saving emergency checkpoint...")
            try:
                current_losses = {
                    "g_total": round(loss_gen_all.item(), 4),
                    "d_total": round(loss_disc.item(), 4)
                }
            except:
                current_losses = {}
            self._save(epoch, tag="latest", losses=current_losses)
            self.logger.info("Saved checkpoint_latest.pth. You can resume with trainer.resume_from_checkpoint()")
            
        finally:
            pbar.close()
            
        self.logger.info("Training Finished.")
