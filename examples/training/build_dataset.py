#!/usr/bin/env python3
"""Build train.txt/val.txt from the multi-speaker dataset.

Combines metadata.txt files from each voice directory into a single
train/val split, with absolute paths to WAV files. No files are copied.

Expected directory layout:
    WAV/
    ├── emilia_dataset/
    │   ├── metadata.txt       # format: relative_wav|speaker_id|lang|text
    │   └── wavs/              # actual audio files
    ├── frieren_dataset/
    │   ├── metadata.txt
    │   └── wavs/
    └── ...

Usage:
    python build_dataset.py --wav-root ./WAV --output-dir ./dataset
"""
import argparse
import json
import random
from pathlib import Path

VOICES = {
    "emilia": 0,
    "frieren": 1,
    "zerotwo": 2,
    "violet": 3,
    "hiro": 4,
}


def read_metadata(voice_dir: Path, voice: str, expected_sid: int):
    """Read metadata.txt and return lines with absolute WAV paths."""
    metadata_path = voice_dir / "metadata.txt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.txt not found: {metadata_path}")

    lines = []
    missing = 0
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"{metadata_path}: unexpected format: {raw_line!r}")
        relative_wav, speaker_id, lang, text = parts
        if int(speaker_id) != expected_sid:
            raise ValueError(
                f"{metadata_path}: speaker_id {speaker_id} != expected {expected_sid} for {voice!r}"
            )
        wav_path = (voice_dir / relative_wav).resolve()
        if not wav_path.is_file():
            missing += 1
            continue
        lines.append(f"{wav_path}|{speaker_id}|{lang}|{text}")
    return lines, missing


def split_train_val(lines: list, val_ratio: float, seed: int):
    """Deterministic per-voice split, always leaving at least 1 example in val."""
    rng = random.Random(seed)
    shuffled = lines[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-root", type=Path, required=True,
                        help="Root directory containing <voice>_dataset/ folders")
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset"),
                        help="Where to write train.txt and val.txt")
    parser.add_argument("--val-ratio", type=float, default=0.01,
                        help="Fraction reserved for validation per voice (default: 1%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lines = []
    val_lines = []
    report = {}

    for voice, speaker_id in VOICES.items():
        voice_dir = args.wav_root / f"{voice}_dataset"
        lines, missing = read_metadata(voice_dir, voice, speaker_id)
        train_split, val_split = split_train_val(lines, args.val_ratio, args.seed)
        train_lines.extend(train_split)
        val_lines.extend(val_split)
        report[voice] = {
            "total": len(lines),
            "missing_wav": missing,
            "train": len(train_split),
            "val": len(val_split),
        }
        if missing:
            print(f"WARNING: {voice} has {missing} line(s) in metadata.txt without matching wav (skipped).")

    # Shuffle the final mix (fixed seed — reproducible)
    rng = random.Random(args.seed)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (args.output_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps({
            "voices": report,
            "total_train": len(train_lines),
            "total_val": len(val_lines),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"train.txt: {len(train_lines)} lines")
    print(f"val.txt:   {len(val_lines)} lines")
    for voice, counts in report.items():
        print(f"  {voice}: {counts['total']} total -> {counts['train']} train / {counts['val']} val")
    print(f"Written to: {args.output_dir}")


if __name__ == "__main__":
    main()
