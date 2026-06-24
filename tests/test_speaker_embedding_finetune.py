import json
import os
import sys
import tempfile
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitsune.data.symbols import symbols
from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.trainer import KitsuneTrainer


def _write_test_wav(path, sample_rate=22050):
    samples = np.arange(2048, dtype=np.float32)
    audio = (0.1 * np.sin(2 * np.pi * 220 * samples / sample_rate) * 32767).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def test_embedding_only_finetune_preserves_shared_weights():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        wav_path = os.path.join(temp_dir, "sample.wav")
        metadata_path = os.path.join(temp_dir, "train.txt")
        config_path = os.path.join(temp_dir, "config.json")
        checkpoint_path = os.path.join(temp_dir, "base.pth")
        output_dir = os.path.join(temp_dir, "output")
        _write_test_wav(wav_path)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(f"{wav_path}|1|pt-br|Olá mundo.\n")

        config = {
            "train": {
                "learning_rate": 1e-3,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": 1,
                "precision": "fp32",
                "use_compile": False,
                "speaker_finetune_mode": "embedding_only",
                "grad_clip": 5.0,
                "segment_size": 64,
                "num_workers": 0,
            },
            "data": {
                "training_files": metadata_path,
                "validation_files": None,
                "sampling_rate": 22050,
                "filter_length": 32,
                "hop_length": 4,
                "win_length": 32,
                "n_mel_channels": 8,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
            },
            "speakers": {"old": 0, "new": 1},
            "model": {
                "inter_channels": 8,
                "hidden_channels": 8,
                "filter_channels": 16,
                "n_heads": 2,
                "n_layers": 1,
                "kernel_size": 3,
                "p_dropout": 0.0,
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_rates": [2, 2],
                "upsample_initial_channel": 16,
                "upsample_kernel_sizes": [4, 4],
                "gin_channels": 4,
                "use_sdp": True,
                "n_speakers": 2,
            },
        }
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file)

        model_params = dict(config["model"])
        model = SynthesizerTrn(
            len(symbols),
            config["data"]["filter_length"] // 2 + 1,
            config["train"]["segment_size"] // config["data"]["hop_length"],
            **model_params,
        )
        torch.save({"generator": model.state_dict(), "step": 0}, checkpoint_path)

        trainer = KitsuneTrainer(config_path, output_dir)
        trainer.resume_from_checkpoint(checkpoint_path)
        eager_model = getattr(trainer.net_g, "_orig_mod", trainer.net_g)
        before = {key: value.detach().clone() for key, value in eager_model.state_dict().items()}
        trainer.fine_tune_speaker_embedding(
            speaker_id=1,
            max_steps=1,
            save_interval=1,
            save_dtype="fp32",
        )
        after = eager_model.state_dict()

        for key, value in before.items():
            if key == "emb_g.weight":
                assert torch.equal(after[key][0], value[0])
            else:
                assert torch.equal(after[key], value), key


if __name__ == "__main__":
    test_embedding_only_finetune_preserves_shared_weights()
    print("Speaker embedding fine-tune test: OK")
