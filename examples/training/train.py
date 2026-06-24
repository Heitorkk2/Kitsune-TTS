#!/usr/bin/env python3
"""Train Kitsune-TTS from scratch or resume from a checkpoint.

Usage:
    # Train from scratch
    python train.py -c config.json -m ./checkpoints

    # Resume from a checkpoint
    python train.py -c config.json -m ./checkpoints -p checkpoints/checkpoint_latest.pth

Requirements:
    - A config.json with model/data/train sections (see model/model_config.json for reference)
    - A dataset with train.txt/val.txt pointing to WAV files (see build_dataset.py)
    - espeak-ng installed (for phonemization)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kitsune.trainer import KitsuneTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Kitsune-TTS")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="JSON file for configuration")
    parser.add_argument("-m", "--model_dir", type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("-p", "--checkpoint", type=str, default=None,
                        help="Path to starting checkpoint (optional)")
    args = parser.parse_args()

    trainer = KitsuneTrainer(config_path=args.config, model_dir=args.model_dir)
    if args.checkpoint:
        trainer.resume_from_checkpoint(args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
