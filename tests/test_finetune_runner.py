"""Small CPU integration checks; no Hub access or GPU/eSpeak installation needed."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np
import torch

from kitsune.api import KitsuneSynthesizer
from kitsune.data.symbols import symbols
from kitsune.finetune.config import FineTuneConfig
from kitsune.finetune.data import BalancedLengthBatchSampler, extract_dataset, file_sha256
from kitsune.finetune.runner import SpeakerFineTuner, load_resume_state, run_finetune
from kitsune.model.synthesizer import SynthesizerTrn


class FakePhonemizer:
    def __init__(self, *args, **kwargs):
        pass

    def phonemize(self, text, lang="pt-br"):
        return "a"


def write_wav(path, frames):
    samples = np.arange(frames, dtype=np.float32)
    audio = (np.sin(samples * 0.04) * 2000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(audio.tobytes())


def make_base(root):
    directory = root / "base"
    directory.mkdir()
    model_config = {
        "model": {
            "inter_channels": 8, "hidden_channels": 8, "filter_channels": 16,
            "n_heads": 2, "n_layers": 1, "kernel_size": 3, "p_dropout": 0.1,
            "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_rates": [16, 16], "upsample_initial_channel": 16,
            "upsample_kernel_sizes": [32, 32], "gin_channels": 4, "use_sdp": True,
        },
        "data": {
            "filter_length": 1024, "hop_length": 256, "win_length": 1024,
            "sampling_rate": 22050, "n_mel_channels": 80, "mel_fmin": 0.0, "mel_fmax": 8000,
        },
        "speakers": {"old_a": 0, "old_b": 1},
    }
    model = SynthesizerTrn(len(symbols), 513, 32, n_speakers=2, **model_config["model"])
    state = {"_orig_mod." + k: v.half() for k, v in model.state_dict().items()}
    torch.save({"generator": state}, directory / "latest_model_fp16.pth")
    (directory / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
    return directory


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_zip_paths_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escaped.txt", "bad")
            config = FineTuneConfig.from_dict({
                "speakers": [{"name": "voice", "wav_dir": "wavs", "transcript": "train.txt"}],
                "dataset_zip": str(archive), "dataset_root": str(root / "data"),
            }, root)
            with self.assertRaises(ValueError):
                extract_dataset(config)
            self.assertFalse((root / "escaped.txt").exists())

    def test_output_cannot_replace_local_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = FineTuneConfig.from_dict({
                "run_name": "voice", "output_root": str(root),
                "model": {"local_dir": str(root / "voice/final")},
                "speakers": [{"name": "voice", "wav_dir": "wavs", "transcript": "train.txt"}],
                "training": {"device": "cpu"},
            })
            with self.assertRaisesRegex(ValueError, "overwrite the base"):
                SpeakerFineTuner(config)

    def test_legacy_numpy_scheduler_state_is_loaded_with_scoped_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            previous = torch.serialization.get_safe_globals().copy()
            torch.save({"optimizer": {"param_groups": [{"lr": np.float64(2e-5)}]},
                        "losses": {"lr_ratio": np.float64(0.5)}}, path)
            payload = load_resume_state(path)
            self.assertEqual(payload["losses"]["lr_ratio"], 0.5)
            self.assertEqual(set(torch.serialization.get_safe_globals()), set(previous))

    def test_sampler_covers_unequal_speakers_and_balances_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for sid, count in ((5, 3), (6, 11)):
                for index in range(count):
                    path = Path(directory) / f"{sid}_{index}.wav"
                    write_wav(path, 1024 * (index + 1))
                    rows.append([str(path), str(sid)])
            dataset = type("Metadata", (), {"audiopaths_and_text": rows})()
            sampler = BalancedLengthBatchSampler(dataset, (5, 6), max_batch_size=8, min_batch_size=2)
            covered = set()
            for batch in sampler:
                covered.update(batch)
                self.assertEqual(sum(rows[i][1] == "5" for i in batch), len(batch) // 2)
                self.assertLessEqual(len(batch), 8)
            self.assertEqual(covered, set(range(len(rows))))

    def test_train_export_resume_and_runtime_compatibility(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("kitsune.data.dataset.EspeakPhonemizer", FakePhonemizer), \
                patch("kitsune.finetune.runner.EspeakPhonemizer", FakePhonemizer), \
                patch("kitsune.api.EspeakPhonemizer", FakePhonemizer):
            root = Path(directory)
            torch.manual_seed(42)
            base = make_base(root)
            before_hash = file_sha256(base / "latest_model_fp16.pth")
            base_state = torch.load(base / "latest_model_fp16.pth", weights_only=True)["generator"]
            wav_dir = root / "wavs"
            wav_dir.mkdir()
            transcript = root / "train.txt"
            for index in range(3):
                write_wav(wav_dir / f"{index}.wav", 12288 + index * 256)
            transcript.write_text("\n".join(f"{i}.wav|A short phrase." for i in range(3)), encoding="utf-8")
            for speaker_count in (1, 2):
                with self.subTest(speaker_count=speaker_count):
                    values = {
                        "run_name": f"voices_{speaker_count}", "output_root": str(root / "output"),
                        "model": {"local_dir": str(base)},
                        "speakers": [{"name": f"new_{i}", "wav_dir": str(wav_dir), "transcript": str(transcript)}
                                     for i in range(speaker_count)],
                        "training": {"max_steps": 2, "warmup_steps": 1, "batch_size": 2,
                                     "min_batch_size": 2, "reference_batch_size": 2,
                                     "grad_accum_steps": 2, "save_interval": 1,
                                     "num_workers": 0, "device": "cpu", "precision": "fp32"},
                    }
                    config = FineTuneConfig.from_dict(values)
                    result = run_finetune(config)
                    final = torch.load(result["checkpoint"], weights_only=True)
                    self.assertEqual(final["adaptation_step"], 2)
                    embedding = final["generator"]["emb_g.weight"]
                    self.assertEqual(embedding.shape[0], 2 + speaker_count)
                    self.assertTrue(torch.equal(embedding[:2], base_state["_orig_mod.emb_g.weight"]))
                    self.assertGreater(int(torch.count_nonzero(embedding[2:])), 0)
                    self.assertTrue(any(
                        not torch.equal(value, base_state["_orig_mod." + key])
                        for key, value in final["generator"].items() if key != "emb_g.weight"
                    ))
                    manifest = json.loads((config.run_dir / "final/training_manifest.json").read_text())
                    self.assertTrue(all(np.isfinite(v) for v in manifest["losses"].values()))
                    self.assertFalse(manifest["old_speakers_guaranteed"])
                    with zipfile.ZipFile(result["archive"]) as archive:
                        self.assertIn("model_config.json", archive.namelist())
                        self.assertIn("run_config.json", archive.namelist())
                        self.assertNotIn("specialized_latest.pth", archive.namelist())
                    synth = KitsuneSynthesizer(checkpoint=result["checkpoint"], config=result["config"], device="cpu")
                    audio = synth.synthesize("A test.", speaker="new_0", noise_scale=0.0)
                    self.assertGreater(len(audio), 0)
                    self.assertTrue(np.isfinite(audio).all())
                    resumed_values = copy.deepcopy(values)
                    resumed_values["resume_state"] = result["training_state"]
                    resumed_values["training"]["max_steps"] = 3
                    resumed = run_finetune(FineTuneConfig.from_dict(resumed_values))
                    resume_payload = torch.load(resumed["training_state"], weights_only=True)
                    self.assertEqual(resume_payload["step"], 3)
                    resumed_values["training"]["max_steps"] = 4
                    resumed_values["speakers"][0]["name"] = "wrong_voice"
                    with self.assertRaisesRegex(ValueError, "names/order"):
                        SpeakerFineTuner(FineTuneConfig.from_dict(resumed_values))
            self.assertEqual(file_sha256(base / "latest_model_fp16.pth"), before_hash)


if __name__ == "__main__":
    unittest.main()
