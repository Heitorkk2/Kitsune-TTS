#!/usr/bin/env python3
"""Profile V1 CPU inference and compare FP32 runtime settings without re-exporting."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kitsune.data.symbols import cleaned_text_to_sequence
from kitsune.phonemizer.phonemizer import EspeakPhonemizer


def options(threads=0, extended=False, spinning=True, parallel=False):
    result = ort.SessionOptions()
    result.intra_op_num_threads = threads
    result.inter_op_num_threads = 2 if parallel else 1
    result.execution_mode = ort.ExecutionMode.ORT_PARALLEL if parallel else ort.ExecutionMode.ORT_SEQUENTIAL
    result.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if extended else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    result.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    result.add_session_config_entry("session.inter_op.allow_spinning", "1" if spinning else "0")
    return result


def profile_summary(events):
    groups, operators, nodes = defaultdict(float), defaultdict(float), defaultdict(float)
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        name = event["name"]
        ms = event.get("dur", 0) / 1000
        group = next((key for key in ("dec", "enc_p", "flow", "dp") if f"/{key}/" in name), "other")
        groups[group] += ms
        operators[event.get("args", {}).get("op_name", "unknown")] += ms
        nodes[name] += ms
    total = sum(groups.values())
    return {
        "groups_percent": {k: round(v * 100 / total, 2) for k, v in groups.items()} if total else {},
        "operators_ms": sorted(operators.items(), key=lambda item: -item[1])[:10],
        "slowest_nodes_ms": sorted(nodes.items(), key=lambda item: -item[1])[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--text", action="append", help="Repeat for multiple real phrases")
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.add_argument("--variant", action="append",
                        choices=("threads4", "threads8", "extended8", "no_spin8", "parallel4x2"),
                        help="Repeat to restrict comparisons; baseline always runs first and last")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    with (args.config or args.model.with_name("model_config.json")).open(encoding="utf-8") as file:
        config = json.load(file)
    texts = args.text or [
        "Olá, eu sou a Frieren.",
        "Um modelo pequeno pode contar grandes histórias. Esta voz está sendo gerada localmente no processador do computador.",
    ]
    phonemizer = EspeakPhonemizer(eager_languages=("pt-br",))
    feeds = []
    for text in texts:
        x = np.asarray([cleaned_text_to_sequence(phonemizer.phonemize(f", {text.strip()} ,", lang="pt-br"))], dtype=np.int64)
        if not x.size:
            parser.error("text produced no phonemes")
        feeds.append({"x": x, "x_lengths": np.asarray([x.shape[1]], dtype=np.int64),
                      "sid": np.asarray([args.speaker_id], dtype=np.int64),
                      "noise_scale": np.asarray([0], dtype=np.float32),
                      "length_scale": np.asarray([1], dtype=np.float32)})
    print(json.dumps({"ort": ort.__version__, "platform": platform.platform(), "texts": texts,
                      "note": "FP32; noise=0; warmup excluded; sequential sessions; phonemization excluded"}), flush=True)
    variants = [
        ("baseline", {}), ("threads4", {"threads": 4}), ("threads8", {"threads": 8}),
        ("extended8", {"threads": 8, "extended": True}),
        ("no_spin8", {"threads": 8, "spinning": False}),
        ("parallel4x2", {"threads": 4, "parallel": True}), ("baseline_repeat", {}),
    ]
    references = []
    for name, settings in variants:
        if args.variant and name not in ("baseline", "baseline_repeat") and name not in args.variant:
            continue
        session = ort.InferenceSession(str(args.model), options(**settings), providers=["CPUExecutionProvider"])
        for index, feed in enumerate(feeds):
            session.run(["audio"], feed)
            durations = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                audio = session.run(["audio"], feed)[0]
                durations.append(time.perf_counter() - started)
            if not np.isfinite(audio).all():
                raise RuntimeError(f"Non-finite waveform for {name}, text {index}")
            if name == "baseline":
                references.append(audio.copy())
            ref = references[index]
            same_shape = audio.shape == ref.shape
            seconds = statistics.median(durations)
            print(json.dumps({"variant": name, "text_index": index, "median_s": round(seconds, 6),
                              "runs_s": durations, "rtf": seconds / (audio.shape[-1] / config["data"]["sampling_rate"]),
                              "bit_equal": np.array_equal(ref, audio),
                              "max_abs_error": float(np.max(np.abs(ref - audio))) if same_shape else None}), flush=True)
        del session
        gc.collect()
    # Profiling adds overhead; never mix its event times into latency benchmarks.
    with tempfile.TemporaryDirectory(prefix="kitsune-ort-profile-") as directory:
        settings = options()
        settings.enable_profiling = True
        settings.profile_file_prefix = str(Path(directory) / "profile")
        session = ort.InferenceSession(str(args.model), settings, providers=["CPUExecutionProvider"])
        session.run(["audio"], feeds[-1])
        profile_path = session.end_profiling()
        del session
        with open(profile_path, encoding="utf-8") as file:
            print(json.dumps({"profile": profile_summary(json.load(file))}), flush=True)


if __name__ == "__main__":
    main()
