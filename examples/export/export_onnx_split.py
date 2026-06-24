#!/usr/bin/env python3
"""Export Kitsune-TTS as separate acoustic and vocoder ONNX graphs.

The split is lossless: it uses the same 39M checkpoint and introduces a graph
boundary immediately before ``Generator``. It is intended for profiling and
for deployments that want to run the vocoder on a different execution
provider.

Usage:
    python examples/export/export_onnx_split.py \
        model/latest_model_fp32.pth \
        model/model_config.json \
        model/kitsune39M_split

Outputs:
    kitsune39M_split_acoustic.onnx
    kitsune39M_split_vocoder.onnx
    kitsune39M_split_manifest.json
"""

import argparse
import json
import os
import sys

import torch
from torch import nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from kitsune.data.symbols import symbols
from kitsune.model.synthesizer import SynthesizerTrn


class KitsuneAcousticONNX(nn.Module):
    """Text, duration and flow graph ending at the vocoder boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, noise_scale, length_scale):
        latent, conditioning, _, _, _ = self.model.infer_latent(
            x,
            x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
        return latent, conditioning


class KitsuneVocoderONNX(nn.Module):
    """HiFi-GAN decoder graph accepting the acoustic graph outputs."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent, conditioning):
        return self.decoder(latent, g=conditioning)


def _extract_generator(checkpoint):
    if "generator" in checkpoint:
        return checkpoint["generator"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _load_model(checkpoint_path, config):
    speakers = config.get("speakers", {})
    if not speakers:
        raise ValueError("Split vocoder export currently requires a multi-speaker model.")

    model_params = dict(config["model"])
    model_params["n_speakers"] = len(speakers)
    model = SynthesizerTrn(
        len(symbols),
        config["data"]["filter_length"] // 2 + 1,
        config.get("train", {}).get("segment_size", 8192)
        // config["data"]["hop_length"],
        **model_params,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in _extract_generator(checkpoint).items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
            f"Examples: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    model.eval()
    model.remove_weight_norm()
    return model, checkpoint


def _legacy_export(module, inputs, output_path, **kwargs):
    try:
        torch.onnx.export(module, inputs, output_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, inputs, output_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Export separate Kitsune acoustic and vocoder ONNX models."
    )
    parser.add_argument("checkpoint", help="Kitsune .pth checkpoint")
    parser.add_argument("config", help="Matching model_config.json")
    parser.add_argument("output_prefix", help="Output path prefix, without extension")
    parser.add_argument("--opset", type=int, default=15)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    output_prefix = os.path.abspath(args.output_prefix)
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    acoustic_path = f"{output_prefix}_acoustic.onnx"
    vocoder_path = f"{output_prefix}_vocoder.onnx"
    manifest_path = f"{output_prefix}_manifest.json"

    model, checkpoint = _load_model(args.checkpoint, config)
    acoustic = KitsuneAcousticONNX(model).eval()
    vocoder = KitsuneVocoderONNX(model.dec).eval()

    text_length = 15
    acoustic_inputs = (
        torch.randint(0, len(symbols), (1, text_length), dtype=torch.int64),
        torch.tensor([text_length], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0.667], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    _legacy_export(
        acoustic,
        acoustic_inputs,
        acoustic_path,
        input_names=["x", "x_lengths", "sid", "noise_scale", "length_scale"],
        output_names=["latent", "conditioning"],
        dynamic_axes={"x": {1: "text_length"}, "latent": {2: "latent_length"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    latent_length = 64
    vocoder_inputs = (
        torch.randn(1, model.inter_channels, latent_length),
        torch.randn(1, model.gin_channels, 1),
    )
    _legacy_export(
        vocoder,
        vocoder_inputs,
        vocoder_path,
        input_names=["latent", "conditioning"],
        output_names=["audio"],
        dynamic_axes={"latent": {2: "latent_length"}, "audio": {2: "audio_length"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    manifest = {
        "format": "kitsune-tts-split-onnx-v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "sample_rate": int(config["data"]["sampling_rate"]),
        "inter_channels": int(model.inter_channels),
        "gin_channels": int(model.gin_channels),
        "speakers": config.get("speakers", {}),
        "acoustic_model": os.path.basename(acoustic_path),
        "vocoder_model": os.path.basename(vocoder_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    acoustic_mb = os.path.getsize(acoustic_path) / (1024 * 1024)
    vocoder_mb = os.path.getsize(vocoder_path) / (1024 * 1024)
    print(f"Acoustic: {acoustic_path} ({acoustic_mb:.1f} MB)")
    print(f"Vocoder : {vocoder_path} ({vocoder_mb:.1f} MB)")
    print(f"Combined: {acoustic_mb + vocoder_mb:.1f} MB")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
