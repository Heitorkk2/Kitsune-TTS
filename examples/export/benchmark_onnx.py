#!/usr/bin/env python3
"""Compare monolithic and split Kitsune ONNX inference on CPU."""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from kitsune.data.symbols import symbols


def _session(path, threads):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _measure(run, iterations, sample_rate):
    run()
    results = []
    last_audio = None
    for _ in range(iterations):
        started = time.perf_counter()
        last_audio = run()
        elapsed = time.perf_counter() - started
        duration = last_audio.shape[-1] / sample_rate
        results.append((elapsed, elapsed / duration))
    return {
        "seconds": statistics.median(item[0] for item in results),
        "rtf": statistics.median(item[1] for item in results),
        "audio": last_audio,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", required=True)
    parser.add_argument("--vocoder", required=True)
    parser.add_argument("--full", help="Optional monolithic ONNX model")
    parser.add_argument("--threads", type=int, default=0, help="0 lets ORT choose")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--text-length", type=int, default=40)
    parser.add_argument("--vocab-size", type=int, default=len(symbols))
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="0 makes the full/split output comparison deterministic",
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.text_length < 1 or args.vocab_size < 2:
        raise ValueError("iterations/text-length must be positive and vocab-size >= 2")

    acoustic = _session(args.acoustic, args.threads)
    vocoder = _session(args.vocoder, args.threads)
    rng = np.random.default_rng(42)
    x = rng.integers(1, args.vocab_size, size=(1, args.text_length), dtype=np.int64)
    feed = {
        "x": x,
        "x_lengths": np.asarray([args.text_length], dtype=np.int64),
        "sid": np.asarray([args.speaker_id], dtype=np.int64),
        "noise_scale": np.asarray([args.noise_scale], dtype=np.float32),
        "length_scale": np.asarray([1.0], dtype=np.float32),
    }

    acoustic_times = []
    vocoder_times = []

    def run_split():
        started = time.perf_counter()
        latent, conditioning = acoustic.run(["latent", "conditioning"], feed)
        acoustic_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        audio = vocoder.run(
            ["audio"], {"latent": latent, "conditioning": conditioning}
        )[0]
        vocoder_times.append(time.perf_counter() - started)
        return audio

    split = _measure(run_split, args.iterations, args.sample_rate)
    # Drop the warmup timings added by _measure().
    acoustic_median = statistics.median(acoustic_times[1:])
    vocoder_median = statistics.median(vocoder_times[1:])
    split_size = (os.path.getsize(args.acoustic) + os.path.getsize(args.vocoder)) / (1024 * 1024)
    print(
        f"split: {split['seconds']:.3f}s, RTF {split['rtf']:.3f}, "
        f"acoustic {acoustic_median:.3f}s, vocoder {vocoder_median:.3f}s, "
        f"{split_size:.1f} MB"
    )

    if args.full:
        full_session = _session(args.full, args.threads)

        def run_full():
            return full_session.run(["audio"], feed)[0]

        full = _measure(run_full, args.iterations, args.sample_rate)
        full_size = os.path.getsize(args.full) / (1024 * 1024)
        print(
            f"full : {full['seconds']:.3f}s, RTF {full['rtf']:.3f}, "
            f"{full_size:.1f} MB"
        )
        if full["audio"].shape == split["audio"].shape:
            max_error = float(np.max(np.abs(full["audio"] - split["audio"])))
            print(f"parity: same shape, max_abs_error={max_error:.8g}")
        else:
            print(
                "parity: different stochastic output lengths "
                f"full={full['audio'].shape}, split={split['audio'].shape}"
            )


if __name__ == "__main__":
    main()
