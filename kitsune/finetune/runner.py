"""Full-generator adaptation extracted from the public Colab notebook.

The training recipe is unchanged: zero new rows, embedding warm-up, then full
generator updates with mel/KL/duration losses. No architecture files are patched.
"""

import json
import os
from pathlib import Path
import random
import shutil
import wave

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from kitsune.data.audio import mel_spectrogram_torch
from kitsune.data.symbols import symbols, cleaned_text_to_sequence
from kitsune.losses import kl_loss
from kitsune.model.commons import slice_segments
from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.phonemizer.phonemizer import EspeakPhonemizer
from .config import FineTuneConfig
from .data import file_sha256, extract_dataset, prepare_datasets, build_loader
from .speakers import expand_speaker, _generator_state


FORMAT = "kitsune-specialized-full-finetune-v1"


def load_resume_state(path):
    """Read old notebook states without enabling unrestricted pickle loading.

    Its cosine scheduler stored NumPy float64 scalars in Adam's learning rates.
    Permit just that numeric representation, including NumPy 1.x/2.x names.
    New states use plain Python floats and need no NumPy globals.
    """
    scalar = np.float64(0).__reduce__()[0]
    allowed = [np.dtype, type(np.dtype("float64")),
               (scalar, "numpy.core.multiarray.scalar"),
               (scalar, "numpy._core.multiarray.scalar")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def write_wav(path, audio, sample_rate):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def fp16_state(model):
    return {
        key: value.detach().cpu().to(torch.float16)
        if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }


class SpeakerFineTuner:
    """Own one run's model, previews and optimizer loop; no notebook globals."""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.work = config.run_dir
        self.train_out = self.work / "training"
        self.final_dir = self.work / "final"
        device = config.training.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        self.device = torch.device(device)
        self.amp_enabled = self.device.type == "cuda" and config.training.precision == "fp16"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self._load_base()
        self._build_model()
        self.phonemizer = None  # Construct lazily; not needed for model-only tests.

    def _load_base(self):
        source = self.config.model
        if source.local_dir:
            self.base_ckpt = Path(source.local_dir) / "latest_model_fp16.pth"
            config_path = Path(source.local_dir) / "model_config.json"
        else:
            from huggingface_hub import hf_hub_download
            directory = Path(self.config.output_root) / "base_model"
            self.base_ckpt = Path(hf_hub_download(
                repo_id=source.repo_id, revision=source.revision,
                filename="latest_model_fp16.pth", local_dir=directory,
            ))
            config_path = Path(hf_hub_download(
                repo_id=source.repo_id, revision=source.revision,
                filename="model_config.json", local_dir=directory,
            ))
        if (self.base_ckpt.resolve() == (self.final_dir / "latest_model_fp16.pth").resolve()
                or config_path.resolve() == (self.final_dir / "model_config.json").resolve()):
            raise ValueError("Output files would overwrite the base model. Choose another run_name/output_root.")
        self.base_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.base_checkpoint = torch.load(self.base_ckpt, map_location="cpu", weights_only=True)
        base_state = _generator_state(self.base_checkpoint)
        key = next(k for k in ("emb_g.weight", "_orig_mod.emb_g.weight") if k in base_state)
        self.base_embedding = base_state[key]
        self.base_speaker_count = len(self.base_config["speakers"])
        if self.base_embedding.shape != (
            self.base_speaker_count, self.base_config["model"]["gin_channels"]
        ):
            raise ValueError("The base checkpoint and speaker configuration do not match.")
        names = [s.name for s in self.config.speakers]
        if set(names).intersection(self.base_config["speakers"]):
            raise ValueError("New speaker names must not overlap with base speakers.")
        self.target_speakers = {
            name: self.base_speaker_count + index for index, name in enumerate(names)
        }
        self.sample_rate = int(self.base_config["data"]["sampling_rate"])
        self.base_sha256 = file_sha256(self.base_ckpt)
        print("Base checkpoint:", self.base_ckpt)
        print("SHA-256:", self.base_sha256)
        print("New speakers:", self.target_speakers)

    def _build_model(self):
        config = self.base_config
        checkpoint = self.base_checkpoint
        placeholder = next(iter(config["speakers"]))
        for name in self.target_speakers:
            config, checkpoint, _ = expand_speaker(config, checkpoint, placeholder, name)
        self.model_config = config
        params = dict(config["model"])
        params["n_speakers"] = len(config["speakers"])
        self.model = SynthesizerTrn(
            len(symbols), config["data"]["filter_length"] // 2 + 1,
            8192 // config["data"]["hop_length"], **params,
        ).to(self.device)
        self.original_state = {
            (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
            for key, value in checkpoint["generator"].items()
        }
        self.model.load_state_dict(self.original_state, strict=True)
        self.embedding = self.model.emb_g.weight
        self.protected_embedding = self.embedding[:self.base_speaker_count].detach().clone()
        self.target_embedding_norm = self.protected_embedding.float().norm(dim=1).median()
        with torch.no_grad():
            self.embedding[self.base_speaker_count:].zero_()
        self.full_model_parameters = [
            p for p in self.model.parameters() if p is not self.embedding
        ]
        self.resume_payload = None
        self.start_step = 0
        if self.config.resume_state:
            payload = load_resume_state(self.config.resume_state)
            if payload.get("format") != FORMAT:
                raise ValueError("Unsupported resume format.")
            if payload.get("base_checkpoint_sha256") != self.base_sha256:
                raise ValueError("Resume state uses a different base checkpoint.")
            if payload.get("speaker_map") != config["speakers"]:
                raise ValueError("Resume state speaker names/order do not match this run.")
            self.start_step = int(payload.get("step", 0))
            if not 0 <= self.start_step < self.config.training.max_steps:
                raise ValueError("max_steps must be greater than the saved update number.")
            self.model.load_state_dict(payload["generator"], strict=True)
            if not torch.equal(self.embedding[:self.base_speaker_count], self.protected_embedding):
                raise ValueError("Resume state modified the protected base embedding rows.")
            self.resume_payload = payload
            print("Resumed specialized training at update", self.start_step)
        self.step = self.start_step
        self.latest_losses = dict(self.resume_payload.get("losses", {}) if self.resume_payload else {})
        self.set_full_finetune(self.start_step >= self.config.training.warmup_steps)
        print("Device:", self.device)
        print("Generator parameters:", sum(p.numel() for p in self.model.parameters()))
        print("New embedding values:", self.embedding[self.base_speaker_count:].numel())

    def set_full_finetune(self, enabled):
        for parameter in self.full_model_parameters:
            parameter.requires_grad_(enabled)
        self.embedding.requires_grad_(True)
        self.model.train(enabled)

    def protect_old_embedding_gradients(self):
        if self.embedding.grad is not None:
            self.embedding.grad[:self.base_speaker_count].zero_()

    @torch.inference_mode()
    def synthesize(self, text, sid, seed=123, noise_scale=0.667):
        if self.phonemizer is None:
            self.phonemizer = EspeakPhonemizer(eager_languages=("pt-br",))
        padded = text.strip()
        if not padded.startswith(","):
            padded = ", " + padded
        if not padded.endswith(","):
            padded += " ,"
        sequence = cleaned_text_to_sequence(self.phonemizer.phonemize(padded, lang=self.config.lang))
        if not sequence:
            raise ValueError("Preview text produced no recognized phonemes.")
        was_training = self.model.training
        self.model.eval()
        try:
            ids = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
            lengths = torch.tensor([ids.size(1)], dtype=torch.long, device=self.device)
            speaker = torch.tensor([sid], dtype=torch.long, device=self.device)
            torch.manual_seed(seed)
            with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.float16):
                audio, *_ = self.model.infer(
                    ids, lengths, sid=speaker, noise_scale=noise_scale, length_scale=1.0,
                )
            return audio[0, 0].float().cpu().numpy()
        finally:
            self.model.train(was_training)

    def save_samples(self, step, directory=None):
        directory = directory or self.train_out / "samples" / f"step_{step:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, sid in self.target_speakers.items():
            write_wav(
                directory / f"{name}.wav",
                self.synthesize(self.config.preview_text, sid, seed=123),
                self.sample_rate,
            )
        return directory

    def save_training_state(self, step, optimizer, scaler, losses):
        path = self.train_out / "specialized_latest.pth"
        temporary = path.with_suffix(".tmp")
        t = self.config.training
        torch.save({
            "format": FORMAT,
            "step": step,
            "base_checkpoint_sha256": self.base_sha256,
            "generator": fp16_state(self.model),
            "speaker_map": self.model_config["speakers"],
            "hf_model_id": self.config.model.repo_id,
            "hf_revision": self.config.model.revision,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "losses": losses,
            "run_config": self.config.to_dict(),
            "hyperparameters": {
                "warmup_steps": t.warmup_steps,
                "embedding_lr": t.embedding_lr,
                "model_lr": t.model_lr,
                "max_physical_batch": t.batch_size,
                "grad_accum_steps": t.grad_accum_steps,
            },
        }, temporary)
        os.replace(temporary, path)
        return path

    def train(self, train_loader):
        """Run the original optimizer-update loop with the same numerical recipe."""
        t = self.config.training
        model = self.model
        embedding = self.embedding
        full_model_parameters = self.full_model_parameters
        protected_embedding = self.protected_embedding
        target_embedding_norm = self.target_embedding_norm
        target_speakers = self.target_speakers
        base_speaker_count = self.base_speaker_count
        config_adapted = self.model_config
        resume_payload = self.resume_payload
        start_step = self.start_step
        set_full_finetune = self.set_full_finetune
        protect_old_embedding_gradients = self.protect_old_embedding_gradients
        save_training_state = self.save_training_state
        save_samples = self.save_samples
        optimizer = torch.optim.Adam(
            [
                {"params": [embedding], "lr": t.embedding_lr},
                {"params": full_model_parameters, "lr": t.model_lr},
            ],
            betas=(0.9, 0.99),
            eps=1e-8,
        )
        scaler = GradScaler(
            "cuda", enabled=self.amp_enabled, init_scale=128.0, growth_interval=500
        )

        if resume_payload is not None:
            if resume_payload.get("optimizer"):
                optimizer.load_state_dict(resume_payload["optimizer"])
            if resume_payload.get("scaler"):
                scaler.load_state_dict(resume_payload["scaler"])

        segment_size = 8192
        step = start_step
        latest_losses = dict(
            resume_payload.get("losses", {}) if resume_payload else {}
        )
        full_finetune_enabled = step >= t.warmup_steps
        set_full_finetune(full_finetune_enabled)
        overflow_streak = 0
        microbatches_accumulated = 0
        metric_sums = {}
        optimizer.zero_grad(set_to_none=True)


        def update_learning_rates(current_step):
            if current_step < t.warmup_steps:
                ratio = 1.0
            else:
                phase = (current_step - t.warmup_steps) / max(
                    1, t.max_steps - t.warmup_steps
                )
                ratio = 0.15 + 0.85 * 0.5 * (
                    1.0 + np.cos(np.pi * min(1.0, phase))
                )
            # Keep resume files weights_only-compatible (NumPy cosine returns a scalar).
            ratio = float(ratio)
            optimizer.param_groups[0]["lr"] = t.embedding_lr * ratio
            optimizer.param_groups[1]["lr"] = t.model_lr * ratio
            return ratio


        progress = tqdm(
            total=t.max_steps,
            initial=min(step, t.max_steps),
            desc="Speaker fine-tune",
        )

        while step < t.max_steps:
            for batch in train_loader:
                if not full_finetune_enabled and step >= t.warmup_steps:
                    full_finetune_enabled = True
                    set_full_finetune(True)
                    optimizer.zero_grad(set_to_none=True)
                    microbatches_accumulated = 0
                    metric_sums = {}
                    print(
                        f"\nWarm-up finished at update {step}; "
                        "full generator unfrozen."
                    )

                lr_ratio = update_learning_rates(step)
                text = batch[0].to(self.device, non_blocking=True)
                text_lengths = batch[1].to(self.device, non_blocking=True)
                spec = batch[2].to(self.device, non_blocking=True)
                spec_lengths = batch[3].to(self.device, non_blocking=True)
                audio = batch[4].to(self.device, non_blocking=True)
                sid = batch[6].to(self.device, non_blocking=True)

                with autocast(
                    device_type=self.device.type,
                    enabled=self.amp_enabled,
                    dtype=torch.float16,
                ):
                    predicted, duration_loss, _, ids_slice, _, z_mask, (
                        _, z_p, m_p, logs_p, _, logs_q
                    ) = model(
                        text, text_lengths, spec, spec_lengths, sid=sid
                    )

                real = slice_segments(
                    audio,
                    ids_slice * config_adapted["data"]["hop_length"],
                    segment_size,
                )
                common = min(real.size(2), predicted.size(2))
                real = real[:, :, :common]
                predicted = predicted[:, :, :common]

                with autocast(device_type=self.device.type, enabled=False):
                    mel_real = mel_spectrogram_torch(
                        real.squeeze(1).float(),
                        config_adapted["data"]["filter_length"],
                        config_adapted["data"]["n_mel_channels"],
                        config_adapted["data"]["sampling_rate"],
                        config_adapted["data"]["hop_length"],
                        config_adapted["data"]["win_length"],
                        config_adapted["data"]["mel_fmin"],
                        config_adapted["data"]["mel_fmax"],
                    )
                    mel_pred = mel_spectrogram_torch(
                        predicted.squeeze(1).float(),
                        config_adapted["data"]["filter_length"],
                        config_adapted["data"]["n_mel_channels"],
                        config_adapted["data"]["sampling_rate"],
                        config_adapted["data"]["hop_length"],
                        config_adapted["data"]["win_length"],
                        config_adapted["data"]["mel_fmin"],
                        config_adapted["data"]["mel_fmax"],
                    )
                    loss_mel = F.l1_loss(mel_pred, mel_real) * 45.0
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_duration = duration_loss.float().mean()

                    new_embeddings = embedding[base_speaker_count:].float()
                    loss_embedding_norm = (
                        (new_embeddings.norm(dim=1) - target_embedding_norm)
                        .pow(2)
                        .mean()
                        * t.embedding_norm_l2
                    )
                    if len(target_speakers) > 1:
                        normalized = F.normalize(new_embeddings, dim=1, eps=1e-8)
                        similarities = normalized @ normalized.T
                        pairs = torch.triu_indices(
                            len(target_speakers), len(target_speakers),
                            offset=1, device=self.device,
                        )
                        loss_diversity = (
                            similarities[pairs[0], pairs[1]].pow(2).mean()
                            * t.embedding_diversity
                        )
                    else:
                        loss_diversity = new_embeddings.new_zeros(())
                    loss = (
                        loss_mel
                        + loss_kl
                        + loss_duration
                        + loss_embedding_norm
                        + loss_diversity
                    )
                    backward_loss = loss / t.grad_accum_steps

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at update {step}: {loss}"
                    )

                scaler.scale(backward_loss).backward()
                protect_old_embedding_gradients()
                microbatches_accumulated += 1

                current_metrics = {
                    "total": float(loss.detach().item()),
                    "mel": float(loss_mel.detach().item()),
                    "kl": float(loss_kl.detach().item()),
                    "duration": float(loss_duration.detach().item()),
                    "embedding_norm": float(loss_embedding_norm.detach().item()),
                    "diversity": float(loss_diversity.detach().item()),
                }
                for name, value in current_metrics.items():
                    metric_sums[name] = metric_sums.get(name, 0.0) + value

                if microbatches_accumulated < t.grad_accum_steps:
                    continue

                scaler.unscale_(optimizer)
                protect_old_embedding_gradients()
                active_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                gradients_finite = all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all().item()
                    for parameter in active_parameters
                )
                if not gradients_finite:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    microbatches_accumulated = 0
                    metric_sums = {}
                    overflow_streak += 1
                    print(
                        f"\nOverflow skipped at update {step}; "
                        f"new scale={scaler.get_scale():.0f}"
                    )
                    if overflow_streak >= 8:
                        raise FloatingPointError(
                            "Repeated non-finite gradients; check precision and learning rates. "
                            "Try FP32 when resuming from a good state."
                        )
                    continue

                overflow_streak = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    active_parameters, 5.0
                )
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    embedding[:base_speaker_count].copy_(protected_embedding)
                optimizer.zero_grad(set_to_none=True)

                step += 1
                latest_losses = {
                    name: value / t.grad_accum_steps
                    for name, value in metric_sums.items()
                }
                latest_losses.update(
                    {
                        "grad_norm": float(grad_norm.detach().item()),
                        "lr_ratio": float(lr_ratio),
                    }
                )
                microbatches_accumulated = 0
                metric_sums = {}

                progress.update(1)
                progress.set_postfix(
                    total=f"{latest_losses['total']:.2f}",
                    mel=f"{latest_losses['mel']:.2f}",
                )

                if step % t.save_interval == 0 or step >= t.max_steps:
                    state_path = save_training_state(
                        step, optimizer, scaler, latest_losses
                    )
                    samples = save_samples(step)
                    print(
                        f"\nstep={step} state={state_path} samples={samples}"
                    )

                if step >= t.max_steps:
                    break

        progress.close()
        print("Training finished:", latest_losses)
        self.step = step
        self.latest_losses = latest_losses

    def verify(self):
        """Check base-file integrity and protected rows, not old-voice identity."""
        if file_sha256(self.base_ckpt) != self.base_sha256:
            raise RuntimeError("The base checkpoint file changed.")
        if not torch.equal(
            self.embedding[:self.base_speaker_count].detach().float(),
            self.protected_embedding.float(),
        ):
            raise RuntimeError("The original embedding rows changed.")
        current = self.model.state_dict()
        changed = sum(
            not torch.equal(current[key].detach().cpu().float(), original.float())
            for key, original in self.original_state.items()
        )
        print("PASS: base checkpoint file and original embedding rows are unchanged.")
        print("Adapted tensors:", changed)
        print("Shared-weight updates may change the original voices in this model.")

    def export(self):
        """Save a standard inference checkpoint, manifest, samples and ZIP."""
        self.verify()
        self.save_samples(self.step, self.final_dir / "samples")
        checkpoint_path = self.final_dir / "latest_model_fp16.pth"
        config_path = self.final_dir / "model_config.json"
        state_path = self.train_out / "specialized_latest.pth"
        generator = fp16_state(self.model)
        generator["emb_g.weight"][:self.base_speaker_count].copy_(self.base_embedding)
        torch.save({
            "generator": generator,
            "step": int(self.base_checkpoint.get("step", 0)),
            "adaptation_step": self.step,
            "epoch": int(self.base_checkpoint.get("epoch", 0)),
            "speaker_finetune": {
                "mode": "specialized_full_generator",
                "new_speakers": self.target_speakers,
                "base_checkpoint_sha256": self.base_sha256,
                "old_speakers_guaranteed": False,
            },
        }, checkpoint_path)
        config_path.write_text(json.dumps(self.model_config, ensure_ascii=False, indent=2), encoding="utf-8")
        t = self.config.training
        manifest = {
            "format": FORMAT,
            "base_checkpoint_sha256": self.base_sha256,
            "hf_model_id": self.config.model.repo_id,
            "hf_revision": self.config.model.revision,
            "checkpoint": checkpoint_path.name,
            "config": config_path.name,
            "training_state": str(state_path),
            "speaker_ids": self.target_speakers,
            "steps": self.step,
            "warmup_steps": t.warmup_steps,
            "max_physical_batch": t.batch_size,
            "min_physical_batch": t.min_batch_size,
            "reference_batch_size": t.reference_batch_size,
            "grad_accum_steps": t.grad_accum_steps,
            "length_bucketing": True,
            "tensor_cache": True,
            "embedding_lr": t.embedding_lr,
            "model_lr": t.model_lr,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "old_speakers_guaranteed": False,
            "base_checkpoint_untouched": True,
            "losses": self.latest_losses,
            "requires_runtime_hook": False,
            "run_config": self.config.to_dict(),
        }
        (self.final_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (self.final_dir / "run_config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        archive = shutil.make_archive(
            str(self.work / f"{self.config.run_name}_specialized"), "zip", str(self.final_dir),
        )
        print("Checkpoint:", checkpoint_path)
        print("Resume state:", state_path, "(outside the ZIP; required to resume)")
        print("ZIP:", archive, f"({os.path.getsize(archive) / 1e6:.1f} MB)")
        return {"checkpoint": str(checkpoint_path), "config": str(config_path),
                "archive": archive, "training_state": str(state_path)}


def run_finetune(config: FineTuneConfig):
    """Shared entry point for CLI, Colab and tests."""
    config.validate()
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    extract_dataset(config)
    for directory in (config.run_dir, config.run_dir / "training", config.run_dir / "final"):
        directory.mkdir(parents=True, exist_ok=True)
    trainer = SpeakerFineTuner(config)
    (config.run_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    rows = prepare_datasets(config, trainer.target_speakers, trainer.sample_rate)
    loader = build_loader(config, trainer.model_config, trainer.target_speakers, rows, trainer.device)
    print("Initial previews:", trainer.save_samples(trainer.start_step))
    trainer.train(loader)
    return trainer.export()
