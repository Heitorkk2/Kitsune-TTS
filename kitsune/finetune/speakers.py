#!/usr/bin/env python3
"""Append a cloned speaker embedding to a Kitsune checkpoint.

Only ``emb_g.weight`` changes shape. The original five embeddings and all
shared 39M-model weights are copied unchanged, including their FP16/FP32 dtype.
"""

import argparse
import json
import os

import torch


def _generator_state(checkpoint):
    if "generator" in checkpoint:
        return checkpoint["generator"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def expand_speaker(config, checkpoint, base_speaker, new_speaker):
    """Return expanded config/checkpoint without mutating either input."""
    config = json.loads(json.dumps(config))
    speakers = config.get("speakers", {})
    base_speaker = base_speaker.strip().lower()
    new_speaker = new_speaker.strip().lower()
    if not new_speaker:
        raise ValueError("New speaker name must not be empty.")
    if base_speaker not in speakers:
        raise ValueError(
            f"Base speaker '{base_speaker}' not found. Available: {list(speakers)}"
        )
    if new_speaker in speakers:
        raise ValueError(f"New speaker '{new_speaker}' already exists.")

    speaker_ids = sorted(int(value) for value in speakers.values())
    if speaker_ids != list(range(len(speaker_ids))):
        raise ValueError(f"Speaker IDs must be contiguous from zero, found {speaker_ids}.")
    base_sid = int(speakers[base_speaker])
    new_sid = len(speaker_ids)

    source_state = _generator_state(checkpoint)
    state_dict = dict(source_state)
    emb_key = next(
        (key for key in ("emb_g.weight", "_orig_mod.emb_g.weight") if key in state_dict),
        None,
    )
    if emb_key is None:
        raise KeyError("Could not find emb_g.weight in the checkpoint.")
    old_embedding = state_dict[emb_key]
    if old_embedding.ndim != 2:
        raise ValueError(f"Expected a 2D speaker embedding, got {old_embedding.shape}.")
    if old_embedding.shape[0] != new_sid:
        raise ValueError(
            "Checkpoint/config speaker mismatch: "
            f"checkpoint has {old_embedding.shape[0]} rows, config has {new_sid} speakers."
        )

    new_embedding = torch.cat(
        [old_embedding, old_embedding[base_sid : base_sid + 1].clone()], dim=0
    )
    state_dict[emb_key] = new_embedding

    config["speakers"][new_speaker] = new_sid
    config["model"]["n_speakers"] = new_sid + 1
    expanded_checkpoint = {
        "generator": state_dict,
        "step": checkpoint.get("step", checkpoint.get("iteration", 0)),
        "epoch": checkpoint.get("epoch", 0),
        "speaker_surgery": {
            "base_speaker": base_speaker,
            "base_speaker_id": base_sid,
            "new_speaker": new_speaker,
            "new_speaker_id": new_sid,
        },
    }
    return config, expanded_checkpoint, emb_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Base FP16/FP32 .pth")
    parser.add_argument("--config", required=True, help="Matching model config JSON")
    parser.add_argument("--base-speaker", required=True, help="Closest existing voice")
    parser.add_argument("--new-speaker", required=True, help="New lowercase speaker name")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, expanded, emb_key = expand_speaker(
        config,
        checkpoint,
        args.base_speaker,
        args.new_speaker,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_surgery.pth")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
    torch.save(expanded, checkpoint_path)

    embedding = expanded["generator"][emb_key]
    surgery = expanded["speaker_surgery"]
    print("Speaker surgery complete")
    print(
        f"Base: {surgery['base_speaker']} (ID {surgery['base_speaker_id']}) -> "
        f"new: {surgery['new_speaker']} (ID {surgery['new_speaker_id']})"
    )
    print(f"Embedding: {tuple(embedding.shape)}, dtype={embedding.dtype}")
    print("Architecture: same ~39M model; only one 256-value row was appended")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
