#!/usr/bin/env python3
"""Generate training configuration and print a detailed model architecture summary.

This matches the notebook flow used during the initial 5090 training run.
"""
import json
import os
import sys

# Ensure kitsune package is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.data.symbols import symbols

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root_dir, "dataset")
    
    # 1. Update relative paths in dataset files to absolute paths
    for txt_file in ["train.txt", "val.txt"]:
        file_path = os.path.join(dataset_path, txt_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    parts = line.strip().split('|')
                    if len(parts) >= 1 and not parts[0].startswith('/') and not ":\\" in parts[0]:
                        parts[0] = os.path.join(dataset_path, parts[0])
                    f.write('|'.join(parts) + '\n')

    train_file = os.path.join(dataset_path, "train.txt")
    val_file = os.path.join(dataset_path, "val.txt")

    # 2. Define multi-voice config
    config = {
        "train": {
            "learning_rate": 2e-4, "betas": [0.8, 0.99], "eps": 1e-9,
            "batch_size": 64,
            "epochs": 10000,
            "log_interval": 50,
            "eval_interval": 1000,
            "save_interval": 1000,
            "precision": "bf16",
            "use_wandb": False,
            "use_compile": True,
            "grad_clip": 5.0,  # Protection against gradient explosion (NaN)
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "seed": 42
        },
        "data": {
            "training_files": train_file,
            "validation_files": val_file,
            "sampling_rate": 22050, "filter_length": 1024, "hop_length": 256,
            "win_length": 1024, "n_mel_channels": 80, "mel_fmin": 0.0,
            "mel_fmax": 8000
        },
        "speakers": {
            "emilia": 0,
            "frieren": 1,
            "zerotwo": 2,
            "violet": 3,
            "hiro": 4
        },
        "model": {
            "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768,
            "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0.1,
            "resblock": "1", "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "upsample_rates": [8, 8, 2, 2], "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4], "gin_channels": 256,
            "use_sdp": True,
            "n_speakers": 5
        }
    }

    config_path = os.path.join(root_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config Multi-Voz salva em: {config_path}")

    # 3. Initialize Model and analyze architecture
    n_vocab = len(symbols)
    n_speakers = len(config["speakers"])

    model = SynthesizerTrn(
        n_vocab,
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"]
    )

    def format_params(num):
        if num > 1e6:
            return f"{num/1e6:.2f} M"
        elif num > 1e3:
            return f"{num/1e3:.2f} K"
        return str(num)

    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = count_params(model)
    vocoder_params = count_params(model.dec)
    text_encoder_params = count_params(model.enc_p)
    posterior_encoder_params = count_params(model.enc_q)
    flow_params = count_params(model.flow)
    duration_predictor_params = count_params(model.dp)
    embedding_params = count_params(model.emb_g) if hasattr(model, 'emb_g') else 0

    print("="*50)
    print("🦊 KITSUNE-TTS NANO - MODEL X-RAY")
    print("="*50)
    print(f"Total Parameters       : {format_params(total_params)}")
    print("-" * 50)
    print("Module Breakdown:")
    print(f"- Generator (Vocoder)  : {format_params(vocoder_params)}")
    print(f"- Text Encoder         : {format_params(text_encoder_params)}")
    print(f"- Posterior Encoder    : {format_params(posterior_encoder_params)}")
    print(f"- Flow (Coupling)      : {format_params(flow_params)}")
    print(f"- Duration Predictor   : {format_params(duration_predictor_params)}")
    print(f"- Speaker Embedding    : {format_params(embedding_params)} ({n_speakers} voices)")
    print("="*50)
    print("📊 TRAINING INFORMATION")
    print("="*50)
    print(f"Batch Size         : {config['train']['batch_size']}")
    print(f"Precision          : {config['train']['precision']}")
    print(f"Target Epochs      : {config['train']['epochs']}")
    print(f"Dataset            : {config['data']['training_files']}")
    print(f"Sample Rate        : {config['data']['sampling_rate']} Hz")
    print("="*50)


if __name__ == "__main__":
    main()
