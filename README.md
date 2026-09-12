<p align="center">
  <img src="assets/logo.png" alt="Kitsune-TTS Logo" width="480"/>
</p>
<p align="center"><em>Ultra-lightweight Text-to-Speech for anime personas and virtual assistants.</em></p>
<p align="center">
  <strong>Language:</strong> Brazilian Portuguese (PT-BR) 🇧🇷
</p>
<p align="center">
  <a href="https://huggingface.co/Heitorkk2/Kitsune-TTS-V1">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-gray?logo=huggingface" alt="Hugging Face: Kitsune-TTS-V1"/>
  </a>
  <a href="https://github.com/Heitorkk2/Kitsune-TTS">
    <img src="https://img.shields.io/badge/GitHub-Repository-gray?logo=github" alt="GitHub: Kitsune-TTS"/>
  </a>
</p>

---

## ✨ What is Kitsune-TTS?

Kitsune-TTS is an **ultra-lightweight, non-autoregressive TTS engine** designed from scratch for:

- 🎀 **Anime / VTuber / Kawaii personas**: character voices that sound expressive and alive.
- ⚡ **CPU-first inference**: real-time synthesis on everyday consumer hardware.
- 🇧🇷 **Portuguese-first**: natively trained on PT-BR phonetic nuances.
- 🪶 **Under 40M parameters**: highly efficient, fast, and compact architecture.

---

## 🎯 Key Features

| Feature | Description |
|:---|:---|
| 🪶 **Ultra-Lightweight** | Under **40M parameters** (~76 MB in FP16 / ~115 MB in ONNX). Fits easily on edge devices. |
| ⚡ **Blazing Fast** | **RTF 0.22** on consumer CPU (4.46× real-time) and **RTF 0.02** on GPU (46× real-time). |
| 🇧🇷 **Portuguese-First** | Natively trained from scratch on Brazilian Portuguese (PT-BR) via `espeak-ng`. |
| 🎭 **Multi-Speaker** | 5 anchor anime personas with smooth voice interpolation (*Voice Walk*). |
| 🛠️ **Zero Quantization** | Pure FP32 precision execution without quality or audio fidelity loss. |
| 🌐 **Multi-Platform** | Native runtimes for **PyTorch**, **ONNX Runtime (CPU/GPU)**, and **JavaScript (Browser/Node)**. |

---

## 📊 Benchmarks & Performance

Measured synthesizing **9.21 seconds of audio** (`noise_scale=0`, speaker *Frieren*, FP32 computation, zero quantization, identical weights):

| Device / Hardware | Backend / Runtime | Checkpoint | Latency | RTF | Real-time Factor |
|:---|:---|:---|:---:|:---:|:---:|
| **Local CPU** <br><sub>AMD Ryzen 7 5700U (8 threads)</sub> | **PyTorch (`fast_cpu`)** ⚡ | FP16 file (~76 MB) | **2.06 s** | **0.224** | **4.46×** |
| **Local CPU** <br><sub>AMD Ryzen 7 5700U (8 threads)</sub> | **PyTorch (`fast_cpu`)** ⚡ | FP32 file (~151 MB) | 2.17 s | 0.235 | 4.25× |
| **Local CPU** <br><sub>AMD Ryzen 7 5700U (8 threads)</sub> | ONNX Runtime CPU | FP32 ONNX (~115 MB) | 2.38 s | 0.259 | 3.86× |
| **Local CPU** <br><sub>AMD Ryzen 7 5700U (8 threads)</sub> | PyTorch (standard) | FP32 file (~151 MB) | 3.84 s | 0.417 | 2.40× |
| **Cloud GPU** <br><sub>Tesla T4 (15 GB)</sub> | **PyTorch (CUDA)** 🚀 | FP16 file (~76 MB) | **0.20 s** | **0.022** | **46.0×** |

> 💡 **Highlights:**
> - **PyTorch `fast_cpu`**: Delivers a **~1.96× speedup** over standard PyTorch CPU, outperforming ONNX Runtime while keeping 100% identical audio fidelity.
> - **GPU Inference**: Generates ~10 seconds of speech in just **200 ms**.
> - Full reproduction scripts and raw data: [BENCHMARK_RESULTS.md](examples/benchmark/BENCHMARK_RESULTS.md).

---

## 🎤 Voice Personas

Kitsune features 5 distinct character voices mapped to speaker IDs `[0-4]`:

| ID | Persona | Origin Reference | Style / Characteristics |
|:---:|:---|:---|:---|
| `0` | **Emilia** | *Re:Zero* | Soft, sweet, and gentle voice |
| `1` | **Frieren** | *Frieren* | Calm, serene, and steady tone |
| `2` | **Zero Two** | *Darling in the Franxx* | Energetic, playful, and expressive |
| `3` | **Violet** | *Violet Evergarden* | Formal, disciplined, and expressive |
| `4` | **Hiro** | *Darling in the Franxx* | Youthful, calm male voice |

---

## 🏗️ Architecture Overview

Built on a **VITS2-Slim** backbone, a pruned, non-autoregressive architecture optimized for:
- **Monotonic Alignment Search (MAS)** for reliable text-to-audio alignment.
- **Log-variance Clamping** `[-15.0, 5.0]` in both `PosteriorEncoder` and `TextEncoder` to prevent KL-loss explosions in low precision.
- **HiFi-GAN v1-lite vocoder** (Spectral Norm disabled in Discriminator for stability).
- **Reduced normalcy flow layers** for speed and minimal memory footprint.

---

## 🚀 Quick Start

### 1. Download Model Weights

Weights are available on [Hugging Face](https://huggingface.co/Heitorkk2/Kitsune-TTS-V1):

```bash
pip install huggingface_hub

# For PyTorch inference (FP16 checkpoint ~76 MB)
hf download Heitorkk2/Kitsune-TTS-V1 latest_model_fp16.pth model_config.json --local-dir model

# For ONNX inference (single file ~115 MB)
hf download Heitorkk2/Kitsune-TTS-V1 kitsune39M.onnx model_config.json --local-dir model
```

*Note: Install `espeak-ng` on your operating system for phonemization.*

---

### 2. Python (PyTorch)

```bash
pip install -e ".[torch]"
```

```python
import scipy.io.wavfile as wavf
from kitsune.api import KitsuneSynthesizer

# Standard PyTorch inference
synth = KitsuneSynthesizer(checkpoint="model/latest_model_fp16.pth")
audio = synth.synthesize("Olá, eu sou a Frieren!", speaker="frieren")
wavf.write("output.wav", 22050, audio)
```

#### ⚡ Fast CPU Vocoder Mode (`fast_cpu=True`)

Enable channels-last 2D convolutions for an immediate **~2× speedup on CPU**:

```python
synth = KitsuneSynthesizer(
    checkpoint="model/latest_model_fp16.pth",
    device="cpu",
    fast_cpu=True,  # 🚀 ~2x faster vocoder execution
)
audio = synth.synthesize("Olá! Como você está?", speaker="frieren")
```

CLI equivalent:
```bash
python infer.py --model model/latest_model_fp16.pth --fast-cpu --text "Olá!" --output output.wav
```

---

### 3. Python (ONNX Runtime)

```bash
pip install -e ".[onnx]"
```

```python
from kitsune.api import KitsuneSynthesizer

synth = KitsuneSynthesizer(onnx_path="model/kitsune39M.onnx")
audio = synth.synthesize("Rodando na CPU com ONNX!", speaker="emilia")
```

---

### 4. JavaScript (Browser / Node.js)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js"></script>
<script src="clients/js/phonemizer.js"></script>
<script src="clients/js/kitsune-tts.js"></script>
<script>
  const tts = new KitsuneTTS();
  await tts.load('./model/kitsune39M.onnx');
  const audio = await tts.synthesize('Olá mundo!', 0);
  tts.play(audio);
</script>
```

---

## 🎨 Fine-Tuning & Custom Speakers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heitorkk2/Kitsune-TTS/blob/main/examples/finetune/add_speakers_colab.ipynb)

Fine-tune new voices using the [Colab notebook](examples/finetune/add_speakers_colab.ipynb) or run locally:

```bash
python finetune.py --config examples/finetune/config.example.json
```

See the [Fine-Tuning Guide](examples/finetune/README.md) for dataset preparation and configuration options. For ONNX export tools, see [Export Guide](examples/export/README.md).

---

## 📂 Project Structure

```
kitsune-tts/
├── kitsune/                     # Core Python engine
│   ├── api.py                   # High-level synthesizer (PyTorch & ONNX)
│   ├── fast_cpu.py              # Channels-last CPU vocoder optimization
│   ├── trainer.py               # Training loop (VITS2 + MPD)
│   ├── model/                   # VITS2-Slim model architecture
│   ├── phonemizer/              # Custom G2P & eSpeak wrapper
│   └── data/                    # Dataset loaders & audio utils
├── clients/
│   └── js/                      # JavaScript ONNX runtime (Web/Node)
├── examples/                    # Usage examples, finetuning & export scripts
└── requirements.txt
```

---

## 📜 License & Credits

This project is licensed under **GPL-3.0**. 

- **Base Architecture:** [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2) (MIT). Used as architectural starting point with transplanted layers; duration predictors, speaker embeddings, and vocab re-initialized from scratch.
- **Phonemization:** eSpeak NG (GPL-3.0), through `phonemizer`.
- **Pretrained Starting Point:** VCTK Corpus (CC BY 4.0).
- **Model weights, code, and synthetic dataset:** Original work, licensed **GPL-3.0**.

---

## 🙏 Acknowledgments

### Special thanks

Special thanks to [Everteson](https://github.com/Everteson) and [Nakamura](https://github.com/NakamuraIA) for helping build the Kitsune-TTS model.

### Inspirations

- [VITS2](https://arxiv.org/abs/2307.16430) — Base architecture inspiration.
- [OmniVoice (k2-fsa)](https://github.com/k2-fsa/OmniVoice) — Zero-shot voice cloning used to bootstrap synthetic training data.
- [XTTS (Coqui)](https://github.com/coqui-ai/TTS) — Synthetic dataset generation reference.
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) — Proof that compact speech models can deliver stunning quality.
- [Piper TTS](https://github.com/rhasspy/piper) — ONNX export and CPU inference reference.

---

<div align="center">

```text
ᓚ₍ ^. ̫ .^₎
```

**Made with ❤️ by [Heitorkk2](https://github.com/Heitorkk2), [Nakamura](https://github.com/NakamuraIA) & [Everteson](https://github.com/Everteson)**

</div>
