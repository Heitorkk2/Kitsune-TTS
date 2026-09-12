# Kitsune-TTS V1 — benchmark results

Measured with the public V1 artifacts, speaker **Frieren**,
`noise_scale=0`, and a fixed PT-BR text that generated **9.21 seconds** of
audio. Latency is the median of warmed model-only runs. `first synthesis` is
the time to receive the complete first waveform after loading; Kitsune-TTS does
not stream samples, so it is not time-to-first-sample.

## Local CPU — AMD Ryzen 7 5700U

Windows 11, Python 3.11.9, PyTorch 2.12.1 CPU, 8 PyTorch threads. Each backend
ran in a fresh Python process for fair memory measurements. Three warmed runs
per case.

| Runtime | Artifact | Median latency | RTF | Real-time factor | First synthesis | RSS after first synthesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | FP32 checkpoint | 3.84 s | 0.417 | 2.40x | 3.97 s | 687 MB |
| PyTorch | FP16 checkpoint file* | 4.05 s | 0.440 | 2.27x | 4.13 s | 658 MB |
| PyTorch + `fast_cpu` | FP32 checkpoint | 2.17 s | 0.235 | 4.25x | 2.39 s | 695 MB |
| PyTorch + `fast_cpu` | FP16 checkpoint file* | **2.06 s** | **0.224** | **4.46x** | **2.36 s** | 665 MB |
| ONNX Runtime CPU | FP32 ONNX | 2.38 s | 0.259 | 3.86x | 2.34 s | **567 MB** |

`fast_cpu` improved PyTorch FP32 from RTF 0.417 to 0.235 (**1.77x**) and
improved the FP16-file path from RTF 0.440 to 0.224 (**1.96x**). It is a
PyTorch CPU-only execution layout; it does not apply to ONNX or JavaScript.

\*The FP16 checkpoint is smaller on disk, but the current Python API executes
both checkpoint files in FP32.

### Local CPU memory and disk details

| Runtime | RAM delta after load | RAM delta after first synthesis | Model artifact | Approx. model + runtime disk footprint** |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | +446 MB | +493 MB | 151 MB | 682 MB |
| PyTorch FP16-file | +364 MB | +463 MB | 76 MB | 607 MB |
| PyTorch `fast_cpu` FP32 | +432 MB | +502 MB | 151 MB | 682 MB |
| PyTorch `fast_cpu` FP16-file | +348 MB | +472 MB | 76 MB | 607 MB |
| ONNX Runtime CPU | +192 MB | +374 MB | 115 MB | 192 MB |

\**Approximate deployment footprint includes the model artifact, Kitsune source
and directly imported package directories. It excludes Python itself, OS/shared
libraries, `espeak-ng`, caches and unrelated dependencies.

## Google Colab — Tesla T4

Colab runtime: Tesla T4 15 GB, PyTorch 2.11.0+cu128, CUDA 12.8. The assigned
CPU was a 2-vCPU Intel Xeon VM, so its CPU numbers are not comparable to the
local Ryzen measurements. Five warmed runs per case.

| Runtime | Artifact | Median latency | RTF | Real-time factor | First synthesis | GPU process memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch CUDA | FP32 checkpoint | 0.207 s | 0.0225 | 44.4x | 1.79 s | 518 MB |
| PyTorch CUDA | FP16 checkpoint file* | **0.200 s** | **0.0218** | **46.0x** | **0.201 s** | 518 MB |

The PyTorch CUDA peak allocation was approximately **254 MB** after first
synthesis. The first FP32 CUDA run includes GPU/runtime warm-up; use warmed
latency for runtime comparison.


## Reproduce

- Local CPU: [benchmark_local_cpu.py](benchmark_local_cpu.py)
- Colab CPU/GPU matrix: [kitsune_cpu_gpu_benchmark_colab.ipynb](kitsune_cpu_gpu_benchmark_colab.ipynb)
- Raw local results: [local_benchmark_isolated/results.csv](local_benchmark_isolated/results.csv)
- Raw Colab results: [kitsune_benchmark_results/results.csv](kitsune_benchmark_results/results.csv)