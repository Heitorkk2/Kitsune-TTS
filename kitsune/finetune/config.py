"""Validated, JSON-serializable settings shared by local runs and Colab."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import re
from typing import List, Optional


@dataclass
class Speaker:
    name: str
    wav_dir: str
    transcript: str


@dataclass
class ModelSource:
    repo_id: str = "Heitorkk2/Kitsune-TTS-V1"
    revision: str = "main"
    local_dir: Optional[str] = None


@dataclass
class Training:
    max_steps: int = 2400
    warmup_steps: int = 100
    batch_size: int = 16
    min_batch_size: int = 4
    reference_batch_size: int = 8
    frame_budget_percentile: int = 95
    grad_accum_steps: int = 2
    embedding_lr: float = 1e-3
    model_lr: float = 2e-5
    embedding_norm_l2: float = 1e-3
    embedding_diversity: float = 1e-2
    precision: str = "fp16"
    save_interval: int = 200
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"


def _construct(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object.")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"Invalid {cls.__name__}: {error}") from error


@dataclass
class FineTuneConfig:
    speakers: List[Speaker]
    run_name: str = "my_voice_run"
    output_root: str = "./finetune_output"
    model: ModelSource = field(default_factory=ModelSource)
    training: Training = field(default_factory=Training)
    dataset_zip: Optional[str] = None
    dataset_root: Optional[str] = None
    resume_state: Optional[str] = None
    lang: str = "pt-br"
    auto_convert: bool = True
    val_ratio: float = 0.02
    preview_text: str = "Olá! Esta é uma demonstração da minha nova voz."

    @classmethod
    def from_dict(cls, values, base_dir=None):
        if not isinstance(values, dict):
            raise ValueError("The run configuration must be a JSON object.")
        values = dict(values)
        if not isinstance(values.get("speakers"), list):
            raise ValueError("speakers must be a list of voice definitions.")
        values["speakers"] = [_construct(Speaker, s) for s in values["speakers"]]
        values["model"] = _construct(ModelSource, values.get("model", {}))
        values["training"] = _construct(Training, values.get("training", {}))
        config = _construct(cls, values)
        config.validate()
        base = Path(base_dir or Path.cwd()).resolve()

        def resolve(value):
            if value is None:
                return None
            path = Path(value).expanduser()
            return str((base / path).resolve())

        config.output_root = resolve(config.output_root)
        config.dataset_root = resolve(config.dataset_root)
        config.dataset_zip = resolve(config.dataset_zip)
        config.resume_state = resolve(config.resume_state)
        config.model.local_dir = resolve(config.model.local_dir)
        for speaker in config.speakers:
            speaker.wav_dir = resolve(speaker.wav_dir)
            speaker.transcript = resolve(speaker.transcript)
        return config

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)

    @property
    def run_dir(self):
        return Path(self.output_root) / self.run_name

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if not isinstance(self.run_name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", self.run_name):
            raise ValueError("run_name must contain only letters, digits, _ or -.")
        if not self.speakers:
            raise ValueError("Configure at least one new speaker.")
        names = [s.name for s in self.speakers]
        if any(not isinstance(n, str) or not re.fullmatch(r"[a-z0-9_-]+", n) for n in names):
            raise ValueError("Speaker names must be lowercase letters, digits, _ or -.")
        if len(names) != len(set(names)):
            raise ValueError("Speaker names must be unique.")
        for value in [self.output_root, *[p for s in self.speakers for p in (s.wav_dir, s.transcript)]]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Output and speaker paths must be non-empty strings.")
        for value in (self.dataset_root, self.dataset_zip, self.resume_state, self.model.local_dir):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Optional paths must be null or non-empty strings.")
        if self.dataset_zip and not self.dataset_root:
            raise ValueError("dataset_root is required when using dataset_zip.")
        for value in (self.model.repo_id, self.model.revision):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model.repo_id and model.revision must be non-empty strings.")
        if self.lang != "pt-br":
            raise ValueError("Kitsune-TTS V1 supports pt-br.")
        if not isinstance(self.preview_text, str) or not self.preview_text.strip():
            raise ValueError("preview_text must not be empty.")
        if type(self.auto_convert) is not bool:
            raise ValueError("auto_convert must be true or false.")
        if type(self.val_ratio) not in (int, float) or not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        t = self.training
        for name in ("max_steps", "warmup_steps", "batch_size", "min_batch_size",
                     "reference_batch_size", "frame_budget_percentile", "grad_accum_steps",
                     "save_interval", "num_workers", "seed"):
            if type(getattr(t, name)) is not int:
                raise ValueError(f"training.{name} must be an integer.")
        if not 0 <= t.warmup_steps < t.max_steps:
            raise ValueError("Require 0 <= warmup_steps < max_steps.")
        if not 0 < t.min_batch_size <= t.batch_size or t.reference_batch_size < 1:
            raise ValueError("Require 0 < min_batch_size <= batch_size and a positive reference_batch_size.")
        if t.batch_size % len(names) or t.min_batch_size % len(names):
            raise ValueError("Min/max batch sizes must be divisible by the number of speakers.")
        if not 0 < t.frame_budget_percentile <= 100:
            raise ValueError("frame_budget_percentile must be in (0, 100].")
        if t.grad_accum_steps < 1 or t.save_interval < 1 or t.num_workers < 0 or t.seed < 0:
            raise ValueError("Invalid accumulation, save interval, worker count or seed.")
        for name in ("embedding_lr", "model_lr", "embedding_norm_l2", "embedding_diversity"):
            value = getattr(t, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"training.{name} must be a finite non-negative number.")
        if t.embedding_lr == 0 or t.model_lr == 0:
            raise ValueError("Learning rates must be positive.")
        if t.precision not in ("fp16", "fp32") or t.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Use precision fp16/fp32 and device auto/cpu/cuda.")
