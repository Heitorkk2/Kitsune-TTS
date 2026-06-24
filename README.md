<p align="center">
  <img src="assets/logo.png" alt="Kitsune-TTS Logo" width="600"/>
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

- 🎀 **Anime / VTuber / Kawaii personas**: character voices that sound alive
- ⚡ **CPU-first inference**: real-time synthesis on local hardware
- 🇧🇷 **Portuguese-first**: natively trained on PT-BR
- 🪶 **Under 40M parameters**: highly efficient, fast, and lightweight architecture

## 🎯 Key Features & Performance

| Feature | Description |
|:--------|:------------|
| **Tiny & Fast** | V1 checkpoint is ~159 MB (FP32) / ~80 MB (FP16) / ONNX export is ~121 MB. On a standard consumer CPU (e.g., Ryzen 5), pure Python inference hits a blazing **RTF of ~0.3** (generating 10 seconds of audio takes less than 3 seconds). |
| **Multi-Speaker** | 5 anchor anime voices with classic speaker embeddings. |
| **Stable Training** | Stabilized for `bfloat16` precision (log-variance clamping, Spectral Norm disengaged). |
| **Voice Walk** | Interpolate between speakers to generate new voice variations. |

## 🏗️ Architecture Overview

Built on a **VITS2-Slim** backbone, a pruned, non-autoregressive architecture optimized for:
- **Monotonic Alignment Search (MAS)** for stable alignment training.
- **Log-variance Clamping** `[-15.0, 5.0]` in both `PosteriorEncoder` and `TextEncoder` to prevent KL-loss explosions in `bf16`.
- **HiFi-GAN v1-lite vocoder** (Spectral Norm disabled in Discriminator for numerical stability during low-precision training).
- **Reduced normalcy flow layers** for speed and smaller memory footprint.

## 🗣️ Supported Languages

**Brazilian Portuguese (PT-BR)** is the exclusive language supported in this `v1` prototype.
Phonemization is handled natively via `espeak-ng`.

*(Native multilingual training is planned for a future `v2`)*

## 🎤 Voice Personas

The model features 5 distinct character voices mapping to speaker IDs `[0-4]`:

| ID | Persona | Character Origin | Style / Characteristics |
|:---|:--------|:-----------------|:------------------------|
| `0` | **Emilia** | *Re:Zero* | Soft, sweet, and gentle voice |
| `1` | **Frieren** | *Frieren* | Calm, serene, and steady voice |
| `2` | **Zero Two** | *Darling in the Franxx* | Energetic, teasing, and playful |
| `3` | **Violet** | *Violet Evergarden* | Formal, structured, and expressive |
| `4` | **Hiro** | *Darling in the Franxx* | Youthful, calm male voice |

## 📂 Project Structure

```
kitsune-tts/
├── kitsune/                     # Core Python package
│   ├── api.py                   # High-level KitsuneSynthesizer (PyTorch + ONNX)
│   ├── trainer.py               # Training loop (VITS2 + MultiPeriodDiscriminator)
│   ├── model/                   # VITS2-Slim architecture
│   ├── phonemizer/              # Custom G2P engine & eSpeak wrapper
│   └── data/                    # Dataset loaders, symbols, and audio utils
├── clients/
│   └── js/                      # JavaScript ONNX runtime (browser / Node.js)
├── examples/                    # Usage examples
└── requirements.txt
```

## 🚀 Quick Start

### Download the model

The V1 weights and matching configuration are published at
[Heitorkk2/Kitsune-TTS-V1](https://huggingface.co/Heitorkk2/Kitsune-TTS-V1).
Run these commands from the project root; local model files go in `model/`.

```bash
pip install huggingface_hub
hf download Heitorkk2/Kitsune-TTS-V1 latest_model_fp16.pth model_config.json --local-dir model
```

For ONNX inference, download `kitsune39M.onnx` and `model_config.json` from the
same repository, or [export it locally](examples/export/README.md):

```bash
hf download Heitorkk2/Kitsune-TTS-V1 kitsune39M.onnx model_config.json --local-dir model
```

Install **espeak-ng** for phonemization before using the Python examples.

### Python (PyTorch)

```bash
pip install -e ".[torch]"
```

```python
import scipy.io.wavfile as wavf
from kitsune.api import KitsuneSynthesizer

synth = KitsuneSynthesizer(checkpoint="model/latest_model_fp16.pth")
print(synth.list_speakers())  # ['emilia', 'frieren', 'zerotwo', 'violet', 'hiro']

audio = synth.synthesize("Olá, eu sou a Frieren!", speaker="frieren")
wavf.write("output.wav", 22050, audio)
```

### Python (ONNX, no GPU needed)

```bash
pip install -e ".[onnx]"
```

```python
from kitsune.api import KitsuneSynthesizer

synth = KitsuneSynthesizer(onnx_path="model/kitsune39M.onnx")
audio = synth.synthesize("Rodando na CPU com ONNX!", speaker="emilia")
```

### JavaScript (Browser)

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

### Add a speaker on Colab

*Fine-tuning is experimental; defaults may evolve as we test more voices.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heitorkk2/Kitsune-TTS/blob/main/examples/finetune/add_speakers_colab.ipynb)

Use the [speaker fine-tuning notebook](examples/finetune/add_speakers_colab.ipynb)
to configure one or more new voices and download the V1 FP16 base automatically.
The notebook is a lightweight launcher for the same config-driven command used
locally:

```bash
python finetune.py --config examples/finetune/config.example.json
```

Edit the example JSON with your dataset paths before running it. Training code
lives in `kitsune/finetune/`, not in notebook cells.

It warms up the new embeddings, then fine-tunes the full generator with AMP FP16,
cached tensors and duration-aware batches. It saves a separate, specialized
checkpoint compatible with the existing Python runtime, with no adapters required.

**Original voices may change in the specialized model.** Keep the untouched base
checkpoint to use those voices. The architecture stays roughly 39M parameters;
each new speaker adds one embedding row. Training time and results depend on the
dataset and available GPU.

The older embedding-only CLI remains available when exact preservation of shared
weights is required. See [the fine-tuning guide](examples/finetune/README.md) for
both workflows.

### Profile the vocoder separately

The alternative split export produces an acoustic ONNX and a vocoder ONNX from
the same 39M checkpoint. See
[`examples/export/README.md`](examples/export/README.md).

## 📜 License

This project is licensed under **GPL-3.0**. This choice reflects both the
project's use of espeak-ng (GPL-3.0) for phonemization, and a deliberate
decision to keep derivative works open.

The Python package, JavaScript client, model code, and published model weights
use the same GPL-3.0 license.

Note: the VITS2 base architecture this project builds on is MIT-licensed
(see Credits below). GPL-3.0 was chosen independently for this project.

## Credits & Third-Party Licenses

- Base model: [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2), MIT. Used as the
  architectural starting point via weight transplant from a VCTK-pretrained checkpoint (text
  encoder + flow + posterior encoder layers carried over; duration predictor and speaker/vocab
  embeddings re-initialized from scratch). This helped avoid the noisy early-training instability
  of starting fully from random weights with limited compute.
  
- Phonemization: espeak-ng (GPL-3.0), via `phonemizer`.
- Initial transplant checkpoint: VCTK Corpus, CC BY 4.0.

Model weights, code, and synthetic dataset are original work, licensed GPL-3.0.

## 🙏 Acknowledgments

### Special thanks

Special thanks to [Everteson](https://github.com/Everteson) for helping build the Kitsune-TTS model.

### Projects that inspired and supported this work

- [VITS2](https://arxiv.org/abs/2307.16430): Base architecture inspiration
- [OmniVoice (k2-fsa)](https://github.com/k2-fsa/OmniVoice): Zero-shot voice cloning used to bootstrap the entire training corpus (no real recordings were used)
- [XTTS (Coqui)](https://github.com/coqui-ai/TTS): Additional synthetic dataset generation
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M): Proof that small parameter counts can achieve incredible quality
- [Piper TTS](https://github.com/rhasspy/piper): ONNX export and CPU inference reference
