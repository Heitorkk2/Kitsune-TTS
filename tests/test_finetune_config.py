import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kitsune.finetune.config import FineTuneConfig


ROOT = Path(__file__).resolve().parents[1]


def minimal_config():
    return {"speakers": [{"name": "my_voice", "wav_dir": "data/wavs", "transcript": "data/train.txt"}]}


class ConfigTests(unittest.TestCase):
    def test_defaults_match_public_notebook(self):
        config = FineTuneConfig.from_dict(minimal_config(), ROOT)
        self.assertEqual(config.model.repo_id, "Heitorkk2/Kitsune-TTS-V1")
        self.assertEqual(config.training.max_steps, 2400)
        self.assertEqual(config.training.warmup_steps, 100)
        self.assertEqual(config.training.batch_size, 16)
        self.assertEqual(config.training.min_batch_size, 4)
        self.assertEqual(config.training.grad_accum_steps, 2)
        self.assertEqual(config.training.embedding_lr, 1e-3)
        self.assertEqual(config.training.model_lr, 2e-5)
        self.assertEqual(config.training.embedding_norm_l2, 1e-3)
        self.assertEqual(config.training.embedding_diversity, 1e-2)
        self.assertEqual(config.training.precision, "fp16")
        self.assertEqual(config.training.save_interval, 200)

    def test_paths_resolve_relative_to_config_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            values = minimal_config()
            values["model"] = {"local_dir": "base"}
            path.write_text(json.dumps(values), encoding="utf-8")
            config = FineTuneConfig.load(path)
            self.assertEqual(Path(config.speakers[0].wav_dir), path.parent / "data/wavs")
            self.assertEqual(Path(config.model.local_dir), path.parent / "base")
            self.assertEqual(FineTuneConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())

    def test_invalid_settings_fail_early(self):
        changes = [
            {"speakers": []}, {"run_name": "../unsafe"}, {"unknown": 1},
            {"model": {"revison": "main"}}, {"training": {"batch_sze": 8}},
            {"training": {"max_steps": 100, "warmup_steps": 100}},
            {"training": {"grad_accum_steps": 0}}, {"training": {"num_workers": -1}},
            {"training": {"embedding_lr": float("nan")}},
            {"training": {"batch_size": True}}, {"training": {"precision": "int8"}},
            {"training": {"min_batch_size": 0}}, {"lang": "en"},
            {"auto_convert": "yes"}, {"val_ratio": 1}, {"dataset_zip": "data.zip"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                FineTuneConfig.from_dict({**minimal_config(), **change})

    def test_duplicate_names_and_unbalanced_batches_rejected(self):
        values = minimal_config()
        values["speakers"] *= 2
        with self.assertRaises(ValueError):
            FineTuneConfig.from_dict(values)
        values["speakers"] = [dict(values["speakers"][0], name=f"voice_{i}") for i in range(3)]
        with self.assertRaises(ValueError):
            FineTuneConfig.from_dict(values)
        values["training"] = {"batch_size": 12, "min_batch_size": 3}
        self.assertEqual(len(FineTuneConfig.from_dict(values).speakers), 3)

    def test_cli_validate_does_not_download_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(minimal_config()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "finetune.py"), "--config", str(path), "--validate-only"],
                cwd=directory, capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout)["training"]["max_steps"], 2400)
            self.assertFalse((Path(directory) / "finetune_output").exists())


if __name__ == "__main__":
    unittest.main()
