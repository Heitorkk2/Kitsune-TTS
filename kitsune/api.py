import json
import os
from typing import Optional, Sequence

import numpy as np

from kitsune.data.symbols import cleaned_text_to_sequence
from kitsune.phonemizer.phonemizer import EspeakPhonemizer


class KitsuneSynthesizer:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        config: Optional[str] = None,
        device: Optional[str] = None,
        onnx_path: Optional[str] = None,
        ort_threads: int = 0,
        providers: Optional[Sequence[str]] = None,
        remove_weight_norm: bool = True,
    ):
        """Initialize a PyTorch or ONNX Kitsune-TTS synthesizer.

        ``ort_threads=0`` lets ONNX Runtime select the native CPU thread count.
        PyTorch is imported only when ``checkpoint`` is used, so ONNX-only
        installations do not need the much larger Torch dependency.
        """
        if checkpoint is None and onnx_path is None:
            raise ValueError("Must provide either checkpoint (PyTorch) or onnx_path (ONNX).")
        if checkpoint is not None and onnx_path is not None:
            raise ValueError("Provide checkpoint or onnx_path, not both.")

        self.backend = "onnx" if onnx_path is not None else "torch"

        if config is None:
            model_path = onnx_path if onnx_path is not None else checkpoint
            ckpt_dir = os.path.dirname(os.path.abspath(model_path))
            model_cfg = os.path.join(ckpt_dir, "model_config.json")
            legacy_cfg = os.path.join(ckpt_dir, "config.json")
            if os.path.isfile(model_cfg):
                config = model_cfg
            elif os.path.isfile(legacy_cfg):
                config = legacy_cfg
            else:
                raise FileNotFoundError(
                    f"No config found in {ckpt_dir}. Provide one via config= argument."
                )

        with open(config, "r", encoding="utf-8") as file:
            self.hparams = json.load(file)

        self.speaker_map = self.hparams.get("speakers", {})
        self.sample_rate = int(self.hparams["data"]["sampling_rate"])
        self.phonemizer = EspeakPhonemizer(eager_languages=("pt-br",))
        self.model = None
        self.ort = None
        self.device = None
        self._torch = None
        self._ort_input_names = frozenset()
        self._ort_output_name = "audio"

        if self.backend == "onnx":
            if ort_threads < 0:
                raise ValueError("ort_threads must be zero (automatic) or a positive integer.")

            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.intra_op_num_threads = int(ort_threads)
            session_options.inter_op_num_threads = 1
            self.ort = ort.InferenceSession(
                onnx_path,
                session_options,
                providers=list(providers) if providers is not None else ["CPUExecutionProvider"],
            )
            self._ort_input_names = frozenset(item.name for item in self.ort.get_inputs())
            output_names = [item.name for item in self.ort.get_outputs()]
            self._ort_output_name = "audio" if "audio" in output_names else output_names[0]
            return

        import torch
        from kitsune.data.symbols import symbols
        from kitsune.model.synthesizer import SynthesizerTrn

        self._torch = torch
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        n_speakers = (
            len(self.speaker_map)
            if self.speaker_map
            else self.hparams.get("model", {}).get("n_speakers", 0)
        )
        model_params = dict(self.hparams["model"])
        model_params["n_speakers"] = n_speakers

        self.model = SynthesizerTrn(
            len(symbols),
            self.hparams["data"]["filter_length"] // 2 + 1,
            self.hparams.get("train", {}).get("segment_size", 8192)
            // self.hparams["data"]["hop_length"],
            **model_params,
        ).to(self.device)

        checkpoint_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if "generator" in checkpoint_dict:
            state_dict = checkpoint_dict["generator"]
        elif "model_state_dict" in checkpoint_dict:
            state_dict = checkpoint_dict["model_state_dict"]
        else:
            state_dict = checkpoint_dict

        clean_state_dict = {
            (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(clean_state_dict)
        self.model.eval()
        if remove_weight_norm:
            self.model.remove_weight_norm()

    def list_speakers(self) -> list:
        """Return the available speaker names."""
        return list(self.speaker_map.keys())

    @staticmethod
    def _pad_text(text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Text must not be empty.")
        if not text.startswith(","):
            text = f", {text}"
        if not text.endswith(","):
            text = f"{text} ,"
        return text

    def _resolve_speaker(self, speaker: str) -> Optional[int]:
        if not self.speaker_map:
            return None
        speaker_key = speaker.lower()
        if speaker_key not in self.speaker_map:
            available = ", ".join(self.list_speakers())
            raise ValueError(f"Speaker '{speaker}' not found. Available: {available}")
        return int(self.speaker_map[speaker_key])

    def _text_to_sequence(self, text: str, lang: str) -> list:
        phonemes = self.phonemizer.phonemize(self._pad_text(text), lang=lang)
        sequence = cleaned_text_to_sequence(phonemes)
        if not sequence:
            raise ValueError("Text produced no recognized phonemes.")
        return sequence

    def synthesize(
        self,
        text: str,
        speaker: str = "frieren",
        lang: str = "pt-br",
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """Synthesize a mono float waveform at the configured sample rate."""
        sid_int = self._resolve_speaker(speaker)
        sequence = self._text_to_sequence(text, lang)

        if self.backend == "torch":
            torch = self._torch
            x = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
            x_lengths = torch.tensor([x.size(1)], dtype=torch.long, device=self.device)
            sid = (
                torch.tensor([sid_int], dtype=torch.long, device=self.device)
                if sid_int is not None
                else None
            )
            with torch.inference_mode():
                audio, _, _, _ = self.model.infer(
                    x,
                    x_lengths,
                    sid=sid,
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                )
            return audio[0, 0].detach().cpu().numpy()

        sequence_array = np.asarray(sequence, dtype=np.int64)[np.newaxis, :]
        feed = {
            "x": sequence_array,
            "x_lengths": np.asarray([sequence_array.shape[1]], dtype=np.int64),
        }
        if "sid" in self._ort_input_names:
            feed["sid"] = np.asarray([sid_int if sid_int is not None else 0], dtype=np.int64)
        if "noise_scale" in self._ort_input_names:
            feed["noise_scale"] = np.asarray([noise_scale], dtype=np.float32)
        if "length_scale" in self._ort_input_names:
            feed["length_scale"] = np.asarray([length_scale], dtype=np.float32)
        if "noise_scale_w" in self._ort_input_names:
            feed["noise_scale_w"] = np.asarray([0.8], dtype=np.float32)
        if "scales" in self._ort_input_names:
            feed["scales"] = np.asarray([noise_scale, length_scale, 0.8], dtype=np.float32)

        audio = self.ort.run([self._ort_output_name], feed)[0]
        return np.asarray(audio[0, 0], dtype=np.float32)

    def voice_walk(
        self,
        text: str,
        speaker_a: str,
        speaker_b: str,
        alpha: float = 0.5,
        lang: str = "pt-br",
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """Synthesize with an interpolation of two PyTorch speaker embeddings."""
        if self.backend == "onnx":
            raise NotImplementedError(
                "Voice walk interpolation is only supported on the PyTorch backend."
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        sid_a_int = self._resolve_speaker(speaker_a)
        sid_b_int = self._resolve_speaker(speaker_b)
        if sid_a_int is None or sid_b_int is None:
            raise ValueError("Voice walk requires a multi-speaker model.")

        sequence = self._text_to_sequence(text, lang)
        torch = self._torch
        x = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        x_lengths = torch.tensor([x.size(1)], dtype=torch.long, device=self.device)
        sid_a = torch.tensor([sid_a_int], dtype=torch.long, device=self.device)
        sid_b = torch.tensor([sid_b_int], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            audio = self.model.voice_walk(
                x,
                x_lengths,
                sid_a,
                sid_b,
                alpha=alpha,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
        return audio[0, 0].detach().cpu().numpy()
