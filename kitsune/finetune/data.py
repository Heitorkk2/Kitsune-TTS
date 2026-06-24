"""Dataset preparation, persistent tensors and balanced duration-aware batches."""

import hashlib
import json
import os
from pathlib import Path
import random
import wave
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from kitsune.data.dataset import KitsuneDataset
from kitsune.data.collate import KitsuneCollate


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_dataset(config):
    """Extract optional uploaded data; never replace an existing dataset directory."""
    if not config.dataset_zip:
        return
    root = Path(config.dataset_root).resolve()
    if root.exists():
        print("Dataset directory already exists; extraction skipped.")
        return
    with zipfile.ZipFile(config.dataset_zip) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        root.mkdir(parents=True)
        archive.extractall(root)


def _audio_ok(path, sample_rate):
    with wave.open(str(path), "rb") as wav_file:
        ok = (
            wav_file.getnchannels() == 1
            and wav_file.getframerate() == sample_rate
            and wav_file.getsampwidth() == 2
        )
        duration = wav_file.getnframes() / wav_file.getframerate()
    return ok, duration


def _convert(source, destination, sample_rate):
    import librosa
    import soundfile
    audio, _ = librosa.load(str(source), sr=sample_rate, mono=True)
    soundfile.write(str(destination), audio, sample_rate, subtype="PCM_16")


def build_dataset(wav_dir, transcript_file, speaker_id, out_dir, sample_rate,
                  lang="pt-br", auto_convert=True, val_ratio=0.02, seed=42):
    wav_dir = Path(wav_dir)
    transcript_file = Path(transcript_file)
    out_dir = Path(out_dir)
    if not transcript_file.is_file():
        raise FileNotFoundError(f"Missing transcript: {transcript_file}")

    converted_dir = out_dir / "wavs_22k"
    available = {path.name.lower(): path for path in wav_dir.glob("*.wav")}
    rows, skipped, total_seconds = [], [], 0.0

    for raw in transcript_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        parts = raw.split("|")
        reference = parts[0].strip().replace("\\", "/").rsplit("/", 1)[-1]
        text = parts[-1].strip()
        if not reference.lower().endswith(".wav"):
            reference += ".wav"
        source = available.get(reference.lower())
        if source is None or not text:
            skipped.append(reference)
            continue

        try:
            ok, duration = _audio_ok(source, sample_rate)
            if not ok:
                if not auto_convert:
                    skipped.append(reference)
                    continue
                converted_dir.mkdir(parents=True, exist_ok=True)
                converted = converted_dir / source.name
                _convert(source, converted, sample_rate)
                source = converted
                ok, duration = _audio_ok(source, sample_rate)
                if not ok:
                    raise ValueError("Conversion did not produce mono PCM16 WAV at the target sample rate")
            rows.append(f"{source.resolve()}|{speaker_id}|{lang}|{text}")
            total_seconds += duration
        except Exception as error:
            skipped.append(f"{reference}: {error}")

    if len(rows) < 2:
        raise RuntimeError(f"At least two valid audio/text pairs are required in {wav_dir}")

    random.Random(seed).shuffle(rows)
    val_count = min(len(rows) - 1, max(1, round(len(rows) * val_ratio)))
    train_rows, val_rows = rows[val_count:], rows[:val_count]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    print(
        f"SID {speaker_id}: {len(rows)} clips, {total_seconds / 60:.1f} min, "
        f"train={len(train_rows)}, val={len(val_rows)}, skipped={len(skipped)}"
    )
    if skipped:
        print("Skipped examples:", skipped[:10])
    return train_rows, val_rows


def prepare_datasets(config, target_speakers, sample_rate):
    joint_dir = config.run_dir / "dataset"
    joint_dir.mkdir(parents=True, exist_ok=True)
    joint_train, joint_val = [], []
    for entry in config.speakers:
        train, val = build_dataset(
            entry.wav_dir, entry.transcript, target_speakers[entry.name],
            joint_dir / entry.name, sample_rate, lang=config.lang,
            auto_convert=config.auto_convert, val_ratio=config.val_ratio,
            seed=config.training.seed,
        )
        joint_train.extend(train)
        joint_val.extend(val)
    random.Random(config.training.seed).shuffle(joint_train)
    random.Random(config.training.seed + 1).shuffle(joint_val)
    (joint_dir / "train.txt").write_text("\n".join(joint_train) + "\n", encoding="utf-8")
    (joint_dir / "val.txt").write_text("\n".join(joint_val) + "\n", encoding="utf-8")
    print("Joint dataset:", len(joint_train), "training /", len(joint_val), "holdout clips")
    return joint_train


class DiskCachedKitsuneDataset(KitsuneDataset):
    """Compute audio/spectrogram tensors once, then reuse them."""

    def __init__(self, metadata_path, cache_dir, **kwargs):
        super().__init__(metadata_path, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index):
        cache_path = self.cache_dir / f"{index:05d}.pt"
        if cache_path.exists():
            try:
                return torch.load(
                    cache_path, map_location="cpu", weights_only=True
                )
            except Exception:
                cache_path.unlink(missing_ok=True)

        item = super().__getitem__(index)
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(item, temporary)
        try:
            os.replace(temporary, cache_path)
        except FileNotFoundError:
            pass
        return item


class BalancedLengthBatchSampler(Sampler):
    """Balance speakers and group similar durations under a frame budget."""

    def __init__(
        self,
        dataset,
        speaker_ids,
        max_batch_size=16,
        min_batch_size=4,
        reference_batch_size=8,
        frame_budget_percentile=95,
        seed=42,
    ):
        if max_batch_size % len(speaker_ids):
            raise ValueError("max_batch_size must be divisible by the number of speakers")
        if min_batch_size % len(speaker_ids):
            raise ValueError("min_batch_size must be divisible by the number of speakers")

        self.speaker_ids = tuple(speaker_ids)
        self.seed = seed
        self.epoch = 0
        self.lengths = {}
        indices = {sid: [] for sid in self.speaker_ids}

        for index, row in enumerate(dataset.audiopaths_and_text):
            sid = int(row[1])
            if sid not in indices:
                continue
            with wave.open(row[0], "rb") as wav_file:
                frames = int(wav_file.getnframes())
            self.lengths[index] = frames
            indices[sid].append(index)

        if any(not values for values in indices.values()):
            raise ValueError(f"A requested speaker is missing from the dataset: {indices}")

        self.sorted_indices = {
            sid: sorted(values, key=self.lengths.__getitem__)
            for sid, values in indices.items()
        }
        all_lengths = np.array(list(self.lengths.values()), dtype=np.float64)
        reference_frames = float(
            np.percentile(all_lengths, frame_budget_percentile)
        )
        self.frame_budget = reference_batch_size * reference_frames
        self.max_per_speaker = max_batch_size // len(self.speaker_ids)
        self.min_per_speaker = min_batch_size // len(self.speaker_ids)
        self.batch_templates = self._build_templates()

    def _build_templates(self):
        templates = []
        cursor = 0
        largest_pool = max(len(values) for values in self.sorted_indices.values())
        # Stretch smaller sorted pools evenly, preserving duration order.
        pools = {
            sid: [
                values[min(len(values) - 1, position * len(values) // largest_pool)]
                for position in range(largest_pool)
            ]
            for sid, values in self.sorted_indices.items()
        }
        while cursor < largest_pool:
            per_speaker = self.max_per_speaker
            while per_speaker > self.min_per_speaker:
                last = min(cursor + per_speaker, largest_pool) - 1
                longest = max(self.lengths[values[last]] for values in pools.values())
                if per_speaker * len(self.speaker_ids) * longest <= self.frame_budget:
                    break
                per_speaker -= 1
            template = {}
            for sid, values in pools.items():
                chunk = values[cursor:cursor + per_speaker]
                # Pad only the final batch with nearby-duration examples.
                while len(chunk) < per_speaker:
                    chunk.append(values[-1])
                template[sid] = chunk
            templates.append(template)
            cursor += per_speaker
        return templates

    def __len__(self):
        return len(self.batch_templates)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.batch_templates)))
        rng.shuffle(order)

        for template_index in order:
            batch = []
            for sid in self.speaker_ids:
                chunk = self.batch_templates[template_index][sid].copy()
                rng.shuffle(chunk)
                batch.extend(chunk)
            rng.shuffle(batch)
            yield batch


def build_loader(config, model_config, target_speakers, joint_train, device):
    metadata_path = (config.run_dir / "dataset") / "train.txt"
    cache_signature = {
        "metadata_sha256": file_sha256(metadata_path),
        "audio_config": model_config["data"],
        "files": [
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in (Path(row.split("|", 1)[0]) for row in joint_train)
        ],
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(cache_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = config.run_dir / "tensor_cache" / cache_fingerprint
    dataset = DiskCachedKitsuneDataset(
        str(metadata_path),
        cache_dir=cache_dir,
        filter_length=model_config["data"]["filter_length"],
        hop_length=model_config["data"]["hop_length"],
        win_length=model_config["data"]["win_length"],
        sampling_rate=model_config["data"]["sampling_rate"],
    )
    batch_sampler = BalancedLengthBatchSampler(
        dataset,
        speaker_ids=tuple(target_speakers.values()),
        max_batch_size=config.training.batch_size,
        min_batch_size=config.training.min_batch_size,
        reference_batch_size=config.training.reference_batch_size,
        frame_budget_percentile=config.training.frame_budget_percentile,
        seed=config.training.seed,
    )
    loader_options = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=KitsuneCollate(),
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    if config.training.num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(**loader_options)

    batch_sizes = [
        sum(len(chunk) for chunk in template.values())
        for template in batch_sampler.batch_templates
    ]
    print("Tensor cache :", cache_dir)
    print("Batches per pass :", len(train_loader))
    print(
        "Physical batch    :",
        f"min={min(batch_sizes)}, median={int(np.median(batch_sizes))}, "
        f"max={max(batch_sizes)}",
    )
    print("The first pass builds the cache; later passes only load tensors.")
    return train_loader
