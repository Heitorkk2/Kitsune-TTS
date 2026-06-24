#!/usr/bin/env python3
"""Export a monolithic Kitsune checkpoint to ONNX.

Usage:
    python3 export_onnx.py <checkpoint.pth> <config.json> <output.onnx>
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.data.symbols import symbols


class KitsuneONNX(nn.Module):
    """Thin wrapper exposing only the inference path.

    ONNX only needs the `infer()` call graph, not the full training-time
    forward pass (posterior encoder, MAS and discriminator), so this module
    re-exposes just that entry point for tracing.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, noise_scale, length_scale):
        audio, _, _, _ = self.model.infer(
            x, x_lengths, sid=sid, noise_scale=noise_scale, length_scale=length_scale
        )
        return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Kitsune .pth checkpoint")
    parser.add_argument("config", help="Matching model config JSON")
    parser.add_argument("output", help="Output .onnx path")
    args = parser.parse_args()
    checkpoint_path, config_path, output_path = args.checkpoint, args.config, args.output
    with open(config_path, encoding="utf-8") as config_file:
        hps = json.load(config_file)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    model_params = dict(hps["model"])
    model_params["n_speakers"] = len(hps["speakers"])

    net_g = SynthesizerTrn(
        len(symbols),
        hps["data"]["filter_length"] // 2 + 1,
        hps.get("train", {}).get("segment_size", 8192) // hps["data"]["hop_length"],
        **model_params,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get(
        "generator",
        checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint)),
    )

    # Checkpoints saved from a torch.compile()-wrapped model have their keys
    # prefixed with "_orig_mod.". Without stripping it, nothing matches the
    # plain (uncompiled) module below and load_state_dict silently leaves
    # every weight at its random init -- the export would run fine but
    # produce silent/garbage audio.
    state_dict = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in state_dict.items()
    }

    missing_keys, unexpected_keys = net_g.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise SystemExit(
            "ABORTED: checkpoint/config mismatch: "
            f"{len(missing_keys)} missing and {len(unexpected_keys)} unexpected weights "
            f"(e.g. missing={missing_keys[:3]}, unexpected={unexpected_keys[:3]})"
        )
    net_g.eval()
    net_g.remove_weight_norm()

    onnx_model = KitsuneONNX(net_g)
    onnx_model.eval()

    # Dummy inputs used purely to trace the graph shape; actual values don't
    # matter, only their dtypes/shapes and which axes are marked dynamic below.
    dummy_inputs = (
        torch.randint(0, len(symbols), (1, 15), dtype=torch.int64),  # x (phoneme ids)
        torch.tensor([15], dtype=torch.int64),                       # x_lengths
        torch.tensor([0], dtype=torch.int64),                        # sid (speaker id)
        torch.tensor([0.667], dtype=torch.float32),                  # noise_scale
        torch.tensor([1.0], dtype=torch.float32),                    # length_scale
    )

    export_kwargs = {
        "input_names": ["x", "x_lengths", "sid", "noise_scale", "length_scale"],
        "output_names": ["audio"],
        "dynamic_axes": {
            # All public clients synthesize one utterance at a time. Keeping
            # batch=1 static lets ORT simplify substantially more shape logic.
            "x": {1: "text_length"},
            "audio": {2: "audio_length"},
        },
        "opset_version": 15,
        "do_constant_folding": True,
    }

    # dynamo=False forces the legacy TorchScript-based tracer. The newer
    # dynamo-based exporter (torch>=2.x default in some versions) doesn't
    # correctly handle VITS's normalizing-flow control flow, so we pin the
    # older path explicitly. Older torch versions don't accept the
    # `dynamo` kwarg at all, hence the fallback.
    try:
        torch.onnx.export(onnx_model, dummy_inputs, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(onnx_model, dummy_inputs, output_path, **export_kwargs)

    print(f"Exported: {output_path} (step {checkpoint.get('step', '?')})")


if __name__ == "__main__":
    main()
