# ONNX exports

The PyTorch API's optional `fast_cpu=True` layout is for CPU inference only.
Export the original checkpoints with the scripts below, not a converted runtime
`state_dict`. The fast CPU option does not apply to ONNX or JavaScript; the 2D
vocoder conversion was slower in local ONNX tests.

Run these commands from the project root. The local weights directory is
`model/` (singular). Use weights and configuration from the same release of
[Heitorkk2/Kitsune-TTS-V1](https://huggingface.co/Heitorkk2/Kitsune-TTS-V1).

```bash
pip install torch onnx onnxruntime huggingface_hub
hf download Heitorkk2/Kitsune-TTS-V1 latest_model_fp32.pth model_config.json --local-dir model
```

If those files are already present locally, skip the download.

## Monolithic model

```bash
python examples/export/export_onnx.py \
  model/latest_model_fp32.pth \
  model/model_config.json \
  model/kitsune39M.onnx
```

This is the single-file export used by the Python and JavaScript clients. It
exports an FP32 inference graph at opset 15, **without quantization**, and does
not modify the checkpoint. The current V1 artifact is approximately 121 MB.
Training-only modules are not part of the inference graph.

The graph has fixed batch size 1 and dynamic text/audio lengths:

| Input | Type | Shape |
| --- | --- | --- |
| `x` | int64 | `[1, text_length]` |
| `x_lengths` | int64 | `[1]` |
| `sid` | int64 | `[1]` |
| `noise_scale` | float32 | `[1]` |
| `length_scale` | float32 | `[1]` |

Output `audio` is float32 `[1, 1, audio_length]`. For deterministic comparisons
against PyTorch, use `noise_scale=0.0` in both runtimes; their random generators
are not identical when noise is enabled.

Check the exported graph:

```bash
python -c "import onnx; onnx.checker.check_model('model/kitsune39M.onnx')"
```

For publishing, upload `kitsune39M.onnx` to the root of the Hugging Face model
repository alongside the matching `model_config.json`, `latest_model_fp16.pth`
and `latest_model_fp32.pth`. Keep the configuration beside the ONNX file when
using the Python client. The exporter only creates local files; it does not
upload anything. Model artifacts in `model/` are ignored by Git.

## CPU profiling without changing the model

```bash
python examples/export/profile_onnx_cpu.py model/kitsune39M.onnx --iterations 5
```

This uses real PT-BR text (requires the Python phonemizer and eSpeak), fixed
speaker ID 1 and `noise_scale=0`. It compares automatic/4/8 threads, graph
optimization levels, worker spinning and parallel execution. Repeat `--text`
to supply your own phrases, or use `--speaker-id` to select another voice.
For a narrower repeat, use `--variant threads8 --variant no_spin8`; the
automatic baseline is still measured before and after the selected variants.

JSON lines report warmed-up inference times, RTF, exact equality and maximum
absolute waveform error against the initial default session. Phonemization and
session initialization are excluded. A final baseline repeat helps reveal
thermal/background-load drift. Compare several runs on an otherwise idle
machine; small differences are not evidence of a reliable speedup. Numerical
closeness alone is not a perceptual quality guarantee.

A separate instrumented run reports time by model component and the slowest
operators/nodes. Its timings include profiling overhead and are not latency
benchmarks. Raw traces use a temporary directory and are removed automatically;
the script does not modify the model or change application defaults.

See [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
for the tested runtime settings.

## Split acoustic model and vocoder

The split uses the same checkpoint and weights. It places the graph boundary
immediately before the HiFi-GAN decoder:

```text
text + speaker -> acoustic.onnx -> latent + conditioning
latent + conditioning -> vocoder.onnx -> waveform
```

```bash
python examples/export/export_onnx_split.py \
  model/latest_model_fp32.pth \
  model/model_config.json \
  model/kitsune39M_split
```

Compare CPU latency, RTF, size, and deterministic output parity:

```bash
python examples/export/benchmark_onnx.py \
  --full model/kitsune39M.onnx \
  --acoustic model/kitsune39M_split_acoustic.onnx \
  --vocoder model/kitsune39M_split_vocoder.onnx \
  --threads 0
```

The split is primarily a profiling/deployment option. It may be slightly slower
on one CPU provider because the latent crosses a session boundary, but it allows
the vocoder—the dominant compute cost—to run on a separate provider such as
WebGPU while the acoustic graph stays on WASM/CPU.
