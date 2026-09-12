#!/usr/bin/env python3
"""Reproducible CPU benchmark for the published Kitsune-TTS V1 artifacts.

Measures process RAM deltas, approximate deployment disk footprint, first complete
synthesis, warmed model-only latency, end-to-end latency and RTF. It does not
measure literal time-to-first-sample because the current API returns full audio.
"""
import argparse
import csv
import gc
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue as queue_module
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kitsune.api import KitsuneSynthesizer


def process_rss_bytes():
    """Current resident memory, with no third-party dependency."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("Psapi.dll")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = ctypes.WinDLL("kernel32").GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    if Path("/proc/self/statm").is_file():
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


def directory_size_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def module_size_bytes(module_name):
    module = importlib.import_module(module_name)
    location = Path(module.__file__).resolve()
    return directory_size_bytes(location.parent) if location.name == "__init__.py" else location.stat().st_size


def system_cpu_name():
    if sys.platform == "win32":
        try:
            name = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if name:
                return name
        except Exception:
            pass
        try:
            name = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-ItemPropertyValue -LiteralPath 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' -Name ProcessorNameString"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if name:
                return name
        except Exception:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine()


def timed(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def model_only(synth, text, speaker, lang, noise_scale, length_scale):
    sequence = synth._text_to_sequence(text, lang)
    speaker_id = synth._resolve_speaker(speaker)
    if synth.backend == "torch":
        x = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([len(sequence)], dtype=torch.long)
        sid = torch.tensor([speaker_id], dtype=torch.long)
        with torch.inference_mode():
            audio = synth.model.infer(x, lengths, sid=sid, noise_scale=noise_scale, length_scale=length_scale)[0]
        return audio[0, 0].numpy()
    x = np.asarray(sequence, dtype=np.int64)[None, :]
    feed = {"x": x, "x_lengths": np.asarray([x.shape[1]], dtype=np.int64)}
    inputs = synth._ort_input_names
    if "sid" in inputs:
        feed["sid"] = np.asarray([speaker_id], dtype=np.int64)
    if "noise_scale" in inputs:
        feed["noise_scale"] = np.asarray([noise_scale], dtype=np.float32)
    if "length_scale" in inputs:
        feed["length_scale"] = np.asarray([length_scale], dtype=np.float32)
    return synth.ort.run([synth._ort_output_name], feed)[0][0, 0]


def disk_footprint(backend, artifact):
    packages = ["numpy", "phonemizer", "num2words"]
    packages.append("torch" if backend == "torch" else "onnxruntime")
    sizes = {"kitsune_source_mb": directory_size_bytes(ROOT / "kitsune") / 2**20}
    for name in packages:
        try:
            sizes[f"package_{name}_mb"] = module_size_bytes(name) / 2**20
        except Exception as error:
            sizes[f"package_{name}_error"] = str(error)
    sizes["artifact_mb"] = Path(artifact).stat().st_size / 2**20
    sizes["approx_model_plus_runtime_mb"] = sum(value for key, value in sizes.items() if key.endswith("_mb"))
    return sizes


def run_case(name, checkpoint=None, onnx_path=None, fast_cpu=False, iterations=5, text="", speaker="frieren",
             lang="pt-br", noise_scale=0.0, length_scale=1.0, threads=8):
    torch.set_num_threads(threads)
    artifact = checkpoint or onnx_path
    backend = "torch" if checkpoint else "onnx"
    row = {"case": name, "backend": backend, "artifact": Path(artifact).name, "fast_cpu": fast_cpu,
           "threads": threads, "status": "ok", "rss_before_mb": (process_rss_bytes() or 0) / 2**20}
    row.update(disk_footprint(backend, artifact))
    synth = None
    try:
        if backend == "onnx":
            synth, row["load_s"] = timed(lambda: KitsuneSynthesizer(
                onnx_path=onnx_path, ort_threads=threads, providers=["CPUExecutionProvider"]))
        else:
            synth, row["load_s"] = timed(lambda: KitsuneSynthesizer(
                checkpoint=checkpoint, device="cpu", fast_cpu=fast_cpu))
        row["rss_after_load_mb"] = (process_rss_bytes() or 0) / 2**20
        row["ram_delta_load_mb"] = row["rss_after_load_mb"] - row["rss_before_mb"]
        torch.manual_seed(2026)
        first_audio, row["first_synthesis_s"] = timed(
            lambda: synth.synthesize(text, speaker=speaker, lang=lang, noise_scale=noise_scale, length_scale=length_scale))
        row["rss_after_first_mb"] = (process_rss_bytes() or 0) / 2**20
        row["ram_delta_first_mb"] = row["rss_after_first_mb"] - row["rss_before_mb"]
        for _ in range(1):
            torch.manual_seed(2026)
            model_only(synth, text, speaker, lang, noise_scale, length_scale)
        model_times, e2e_times = [], []
        for _ in range(iterations):
            torch.manual_seed(2026)
            audio, elapsed = timed(lambda: model_only(synth, text, speaker, lang, noise_scale, length_scale))
            model_times.append(elapsed)
            torch.manual_seed(2026)
            _, elapsed = timed(lambda: synth.synthesize(text, speaker=speaker, lang=lang, noise_scale=noise_scale, length_scale=length_scale))
            e2e_times.append(elapsed)
        if not np.isfinite(first_audio).all() or not np.isfinite(audio).all():
            raise RuntimeError("non-finite waveform")
        row.update({"audio_s": len(audio) / synth.sample_rate, "model_median_s": statistics.median(model_times),
                    "e2e_median_s": statistics.median(e2e_times), "model_runs_s": model_times,
                    "e2e_runs_s": e2e_times, "rss_after_benchmark_mb": (process_rss_bytes() or 0) / 2**20})
        row["rtf"] = row["model_median_s"] / row["audio_s"]
        row["first_rtf"] = row["first_synthesis_s"] / row["audio_s"]
        return row
    except Exception as error:
        row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        return row
    finally:
        del synth
        gc.collect()


def _case_worker(queue, kwargs):
    """Worker target: one fresh Python process gives each case an isolated RSS."""
    try:
        queue.put(run_case(**kwargs))
    except BaseException as error:
        queue.put({"case": kwargs["name"], "status": "failed", "error": f"worker {type(error).__name__}: {error}"})


def run_case_isolated(**kwargs):
    """Run a case in a clean spawned process so previous allocators cannot skew RAM."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_case_worker, args=(queue, kwargs))
    worker.start()
    worker.join(timeout=900)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        return {"case": kwargs["name"], "status": "failed", "error": "worker exceeded 900 seconds"}
    try:
        return queue.get(timeout=5)
    except queue_module.Empty:
        return {"case": kwargs["name"], "status": "failed", "error": f"worker exited with code {worker.exitcode}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--speaker", default="frieren")
    parser.add_argument("--text", default="Olá! Hoje eu vou contar uma pequena história. A noite estava tranquila, mas uma luz apareceu entre as árvores. Você também conseguiu enxergar?")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "local_benchmark_output")
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 1:
        parser.error("threads and iterations must be positive")
    torch.set_num_threads(args.threads)
    model_dir = args.model_dir.resolve()
    fp32, fp16, onnx = (model_dir / "latest_model_fp32.pth", model_dir / "latest_model_fp16.pth", model_dir / "kitsune39M.onnx")
    required = [fp32, fp16, model_dir / "model_config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required model artifacts: " + ", ".join(missing))
    hardware = {"cpu": system_cpu_name(), "logical_cpus": os.cpu_count(), "platform": platform.platform(),
                "python": sys.version, "torch": torch.__version__, "cpu_threads": args.threads,
                "text": args.text, "speaker": args.speaker, "noise_scale": 0.0}
    print(json.dumps(hardware, indent=2), flush=True)
    cases = [("pytorch_fp32_cpu", dict(checkpoint=str(fp32))), ("pytorch_fp16_file_cpu", dict(checkpoint=str(fp16))),
             ("pytorch_fp32_fast_cpu", dict(checkpoint=str(fp32), fast_cpu=True)),
             ("pytorch_fp16_file_fast_cpu", dict(checkpoint=str(fp16), fast_cpu=True))]
    if onnx.is_file():
        cases.append(("onnx_cpu", dict(onnx_path=str(onnx))))
    rows = []
    for name, options in cases:
        print("Running", name, flush=True)
        row = run_case_isolated(name=name, iterations=args.iterations, text=args.text, speaker=args.speaker,
                                threads=args.threads, **options)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    columns = sorted({key for row in rows for key in row})
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
