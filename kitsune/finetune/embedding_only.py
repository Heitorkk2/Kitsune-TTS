#!/usr/bin/env python3
"""Safely add a voice to Kitsune-TTS on a Colab T4.

This script adapts only the new 256-value speaker embedding. Every shared 39M
model weight and every existing speaker row remains frozen, so old voices are
preserved. The forward/backward pass uses AMP FP16 on CUDA, while the embedding
optimizer keeps an FP32 master value for stability.

Run ``add_speaker.py`` first, then:

    python examples/finetune/finetune.py \
        --config ./new_voice/config.json \
        --checkpoint ./new_voice/checkpoint_surgery.pth \
        --dataset-dir ./new_voice/dataset \
        --output-dir ./new_voice/output \
        --speaker marcelo \
        --max-steps 2000

The final model is still the same architecture (roughly 39M parameters), with
one additional 256-value speaker embedding row.
"""

import argparse
import copy
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from kitsune.trainer import KitsuneTrainer


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_complete_config(config_path):
    """Allow model_config.json or a complete training config as input."""
    defaults_path = os.path.join(REPO_ROOT, "configs", "kitsune_base.json")
    with open(defaults_path, encoding="utf-8") as defaults_file:
        config = json.load(defaults_file)
    with open(config_path, encoding="utf-8") as config_file:
        return _deep_update(config, json.load(config_file))


def _resolve_speaker(config, speaker_name, speaker_id):
    speakers = config.get("speakers", {})
    if speaker_name is not None:
        normalized = speaker_name.strip().lower()
        if normalized not in speakers:
            raise ValueError(
                f"Speaker '{speaker_name}' not found. Available: {list(speakers)}"
            )
        return normalized, int(speakers[normalized])

    matches = [name for name, sid in speakers.items() if int(sid) == speaker_id]
    if not matches:
        raise ValueError(
            f"Speaker ID {speaker_id} not found. Available mapping: {speakers}"
        )
    return matches[0], int(speaker_id)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Embedding-only Kitsune speaker adaptation for Colab/T4."
    )
    parser.add_argument("--config", required=True, help="Surgery/model config JSON")
    parser.add_argument("--checkpoint", required=True, help="Expanded speaker checkpoint")
    parser.add_argument("--dataset-dir", required=True, help="Directory with train.txt")
    parser.add_argument("--output-dir", default="./finetune_output")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--speaker", help="New speaker name from the config")
    target.add_argument("--speaker-id", type=int, help="New speaker ID (name is safer)")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Embedding learning rate")
    parser.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16")
    parser.add_argument("--save-dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--embedding-l2", type=float, default=1e-4)
    parser.add_argument(
        "--resume-delta",
        help="Optional speaker_embedding_delta.pth from an interrupted Colab run",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        raise ValueError("batch-size/lr must be positive and num-workers non-negative")

    config = _load_complete_config(args.config)
    speaker_name, speaker_id = _resolve_speaker(
        config, args.speaker, args.speaker_id
    )
    newest_speaker_id = max(int(value) for value in config["speakers"].values())
    if speaker_id != newest_speaker_id:
        raise ValueError(
            "Safe adaptation only accepts the newly appended (last) speaker. "
            f"Expected ID {newest_speaker_id}, received {speaker_id}."
        )
    train_path = os.path.abspath(os.path.join(args.dataset_dir, "train.txt"))
    validation_path = os.path.abspath(os.path.join(args.dataset_dir, "val.txt"))
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training metadata not found: {train_path}")

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = (
        validation_path if os.path.isfile(validation_path) else None
    )
    config["train"].update(
        {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "use_compile": False,
            "speaker_finetune_mode": "embedding_only",
            "num_workers": args.num_workers,
            # The custom loop is step-bounded; this value is metadata only.
            "epochs": 1,
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    patched_config_path = os.path.join(args.output_dir, "finetune_config.json")
    with open(patched_config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    print(f"Target speaker : {speaker_name} (ID {speaker_id})")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Precision      : {args.precision} AMP")
    print(f"Trainable      : 256 speaker-embedding values")
    print(f"Steps / batch  : {args.max_steps} / {args.batch_size}")
    print("Old voices     : shared weights and old embedding rows frozen")

    trainer = KitsuneTrainer(
        config_path=patched_config_path,
        model_dir=args.output_dir,
    )
    trainer.resume_from_checkpoint(args.checkpoint)
    if args.resume_delta:
        import torch

        delta = torch.load(args.resume_delta, map_location="cpu", weights_only=True)
        if delta.get("format") != "kitsune-speaker-embedding-delta-v1":
            raise ValueError("Unsupported speaker delta format.")
        if int(delta["speaker_id"]) != speaker_id:
            raise ValueError(
                f"Delta targets speaker ID {delta['speaker_id']}, expected {speaker_id}."
            )
        model = getattr(trainer.net_g, "_orig_mod", trainer.net_g)
        target = model.emb_g.weight[speaker_id]
        if tuple(delta["embedding"].shape) != tuple(target.shape):
            raise ValueError(
                f"Delta shape {tuple(delta['embedding'].shape)} != {tuple(target.shape)}."
            )
        with torch.no_grad():
            target.copy_(delta["embedding"].to(device=target.device, dtype=target.dtype))
        print(
            f"Resumed embedding delta from adaptation step "
            f"{delta.get('adaptation_step', '?')}"
        )
    model_path, delta_path = trainer.fine_tune_speaker_embedding(
        speaker_id=speaker_id,
        max_steps=args.max_steps,
        embedding_l2=args.embedding_l2,
        save_interval=args.save_interval,
        save_dtype=args.save_dtype,
    )
    print(f"Model saved     : {model_path}")
    print(f"Embedding delta : {delta_path}")


if __name__ == "__main__":
    main()
