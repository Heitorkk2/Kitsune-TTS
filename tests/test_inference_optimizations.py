"""CPU regression tests for compact deployment and gradient reductions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kitsune.api import KitsuneSynthesizer
from kitsune.fast_cpu import ChannelsLastVocoder
from kitsune.data.symbols import symbols
from kitsune.finetune.runner import gradients_are_finite
from kitsune.model.synthesizer import SynthesizerTrn


class InferenceOptimizationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_checkpoint_formats_and_compact_parity(self):
        cfg = {
            "model": {
                "inter_channels": 8, "hidden_channels": 8, "filter_channels": 16,
                "n_heads": 2, "n_layers": 1, "kernel_size": 3, "p_dropout": 0.1,
                "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_rates": [2, 2], "upsample_initial_channel": 16,
                "upsample_kernel_sizes": [4, 4], "gin_channels": 4,
            },
            "data": {"filter_length": 16, "hop_length": 4, "sampling_rate": 22050},
            "speakers": {"a": 0, "b": 1},
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("kitsune.api.EspeakPhonemizer") as phonemizer:
            phonemizer.return_value.phonemize.return_value = "aaa"
            root = Path(directory)
            config = root / "model_config.json"
            checkpoint = root / "model.pth"
            for sdp in (True, False):
                cfg["model"]["use_sdp"] = sdp
                config.write_text(json.dumps(cfg), encoding="utf-8")
                model = SynthesizerTrn(len(symbols), 9, 8, n_speakers=2, **cfg["model"])
                state = model.state_dict()
                for wrapper in ("model", "generator", "model_state_dict", None):
                    with self.subTest(sdp=sdp, wrapper=wrapper):
                        prefixed = {"_orig_mod." + k: v for k, v in state.items()}
                        torch.save({wrapper: prefixed} if wrapper else prefixed, checkpoint)
                        full = KitsuneSynthesizer(checkpoint=str(checkpoint), compact=False, device="cpu")
                        compact = KitsuneSynthesizer(checkpoint=str(checkpoint), device="cpu")
                        fast = KitsuneSynthesizer(checkpoint=str(checkpoint), fast_cpu=True)
                        self.assertEqual(fast.device.type, "cpu")
                        self.assertIsInstance(fast.model.dec, ChannelsLastVocoder)
                        self.assertFalse(hasattr(compact.model, "enc_q"))
                        self.assertLess(sum(p.numel() for p in compact.model.parameters()),
                                        sum(p.numel() for p in full.model.parameters()))
                        for noise in (0.0, 0.667):
                            torch.manual_seed(123)
                            expected = full.synthesize("test", speaker="a", noise_scale=noise)
                            torch.manual_seed(123)
                            actual = compact.synthesize("test", speaker="a", noise_scale=noise)
                            np.testing.assert_array_equal(actual, expected)
                            torch.manual_seed(123)
                            optimized = fast.synthesize("test", speaker="a", noise_scale=noise)
                            np.testing.assert_allclose(optimized, expected, atol=1e-5, rtol=1e-4)
                        torch.manual_seed(123)
                        expected = full.voice_walk("test", "a", "b", noise_scale=0)
                        torch.manual_seed(123)
                        actual = compact.voice_walk("test", "a", "b", noise_scale=0)
                        np.testing.assert_array_equal(actual, expected)
                        torch.manual_seed(123)
                        optimized = fast.voice_walk("test", "a", "b", noise_scale=0)
                        np.testing.assert_allclose(optimized, expected, atol=1e-5, rtol=1e-4)
                        with self.assertRaisesRegex(ValueError, "device='cpu'"):
                            KitsuneSynthesizer(checkpoint=str(checkpoint), device="cuda", fast_cpu=True)
                # Compact loading must not hide a damaged checkpoint.
                incomplete = dict(state)
                del incomplete["enc_q.pre.weight"]
                torch.save({"model": incomplete}, checkpoint)
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    KitsuneSynthesizer(checkpoint=str(checkpoint), device="cpu")

    def test_fast_cpu_rejects_incompatible_options_early(self):
        with self.assertRaisesRegex(ValueError, "not ONNX"):
            KitsuneSynthesizer(onnx_path="unused.onnx", fast_cpu=True)
        with self.assertRaisesRegex(ValueError, "remove_weight_norm"):
            KitsuneSynthesizer(checkpoint="unused.pth", fast_cpu=True, remove_weight_norm=False)

    def test_conversion_preserves_rng_and_coefficients(self):
        conv = torch.nn.Sequential(torch.nn.Conv1d(4, 8, 3), torch.nn.ConvTranspose1d(8, 4, 4, stride=2))
        rng = torch.get_rng_state().clone()
        converted = ChannelsLastVocoder(conv)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for old, new in zip(conv, converted.decoder):
            self.assertTrue(torch.equal(old.weight, new.weight.squeeze(2)))
            self.assertTrue(torch.equal(old.bias, new.bias))
            self.assertIsInstance(old, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))

    def test_gradient_reduction_preserves_values_and_handles_nonfinite(self):
        parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(3)]
        self.assertTrue(gradients_are_finite(parameters))
        self.assertTrue(gradients_are_finite([]))
        for p in parameters:
            p.grad = torch.arange(3, dtype=torch.float32)
        self.assertTrue(gradients_are_finite(parameters))
        for p in parameters:
            torch.testing.assert_close(p.grad, torch.arange(3, dtype=torch.float32))
        for bad in (float("nan"), float("inf"), -float("inf")):
            parameters[-1].grad[0] = bad
            self.assertFalse(gradients_are_finite(parameters))


if __name__ == "__main__":
    unittest.main()
