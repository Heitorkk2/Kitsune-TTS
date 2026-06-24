# Add speakers to Kitsune-TTS

*Fine-tuning is experimental; defaults may evolve as we test more voices.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heitorkk2/Kitsune-TTS/blob/main/examples/finetune/add_speakers_colab.ipynb)

Use [the notebook](add_speakers_colab.ipynb) for Colab, or `finetune.py` at the
project root for local training. Both run the **same config-driven implementation**.

The notebook only installs dependencies, writes your configuration, starts the
command and lets you listen/download. There is no separate training loop to keep
in sync.

## Local quick start

Install eSpeak NG and the Python dependencies, then edit
[config.example.json](config.example.json) with your speaker names and paths:

```bash
pip install -r requirements.txt
python finetune.py --config examples/finetune/config.example.json --validate-only
python finetune.py --config examples/finetune/config.example.json
```

`--validate-only` checks the configuration and prints effective settings/resolved
paths; it does not download weights, inspect audio, or train. Unknown keys and
invalid values fail early.

Relative paths are resolved **against the JSON file's directory**, not your shell's
working directory. The run config is not the base `model_config.json`.

## What you configure

- `speakers`: one or more objects with `name`, `wav_dir` and `transcript`.
  Names are unique lowercase identifiers. IDs are assigned automatically.
- `run_name` / `output_root`: use a separate name for each independent run.
- `model`: defaults to FP16 weights + configuration from
  [Heitorkk2/Kitsune-TTS-V1](https://huggingface.co/Heitorkk2/Kitsune-TTS-V1).
  Set `local_dir` to use local `latest_model_fp16.pth` and `model_config.json`.
  Set `revision` to a fixed Hub commit for repeatability.
- `training`: override only the settings you want; the rest use shared defaults.
- Optional `dataset_zip` + `dataset_root`: extract uploaded data if the destination
  does not already exist.
- Optional `resume_state`: a saved `training/specialized_latest.pth`.

Transcripts use UTF-8 `filename.wav|text` lines. Existing four-field metadata is
also accepted (first/last fields are used). Do not include `|` in the text.
WAVs can be converted to mono PCM16 at the model's sample rate; originals remain
untouched. A small holdout is excluded from training, but no validation metric is
computed automatically.

## Training defaults

New embedding rows start at zero. After 100 embedding-only warm-up updates, the
full generator is unfrozen. The recipe remains mel reconstruction + KL + duration,
with small embedding regularizers; no discriminator is trained.

| Setting | Default |
| --- | --- |
| Optimizer updates | `max_steps: 2400` |
| Physical batches | `batch_size: 16`, `min_batch_size: 4` |
| Duration budget | `reference_batch_size: 8`, `frame_budget_percentile: 95` |
| Accumulation | `grad_accum_steps: 2` |
| Learning rates | `embedding_lr: 1e-3`, `model_lr: 2e-5` |
| Precision / saves | `precision: "fp16"`, `save_interval: 200` |

The complete defaults live in [config.py](../../kitsune/finetune/config.py).
AMP FP16 uses FP32 parameters and optimizer state. Min/max batch sizes must be
divisible by the number of new voices. For low VRAM, lower the frame budget and
minimum batch size. More accumulation does not inherently make training faster.
Set `training.device` to `auto` (default), `cpu` or `cuda`.

**Original voices may change in the specialized model.** Their embedding rows are
preserved, but shared weights are trained. Keep the untouched base checkpoint for
the original voices. The architecture stays roughly 39M, with one additional
embedding row per new speaker and no runtime adapter.

V1 supports Brazilian Portuguese. Use recordings you have permission to use.
Colab time, memory and final voice quality depend on the dataset/hardware; there
is no guaranteed runtime or quality threshold.

## Results and resume

Outputs under `output_root/run_name/`:

```text
run_config.json
training/
  specialized_latest.pth       # FP16 model + Adam/scaler state; keep to resume
  samples/step_0200/           # Intermediate previews
final/
  latest_model_fp16.pth        # Standard inference checkpoint
  model_config.json
  training_manifest.json
  run_config.json
  samples/
<run_name>_specialized.zip     # Contents of final/ only
```

Dataset lists and tensor caches are also stored inside the run directory.
The latest training state is replaced atomically at each save interval. Download
it separately from the inference ZIP before Colab storage disappears.

To continue, restore the same data/base revision and speaker names/order, set
`resume_state`, and increase `training.max_steps` beyond the saved update.
Resume files from the previous public notebook are supported. FP16 rounding and
recreated data-loader/RNG state mean resume is not bit-exact; changing the total
step budget also changes the cosine schedule.

The final checkpoint loads with the normal `KitsuneSynthesizer` and its matching
config. See [ONNX export](../export/README.md) for deployment.

## Maintainer layout

- [Root finetune.py](../../finetune.py): public JSON CLI.
- [kitsune/finetune/](../../kitsune/finetune/): configuration, data, training and
  checkpoint helpers, shared by local/Colab runs.
- `examples/finetune/*.py`: small compatibility entry points for the older
  embedding-only workflow. Existing commands and `expand_speaker` imports still work.

<details>
<summary>Legacy embedding-only CLI: frozen shared weights</summary>

The following CLI is a separate, more conservative workflow; it is **not** the
full-generator recipe used by the notebook above.

This workflow keeps the Kitsune architecture at roughly 39M parameters. Adding
a speaker appends only one 256-value row to `emb_g.weight`.

The default fine-tune mode is deliberately conservative:

- all shared generator weights are frozen;
- all existing speaker rows are restored exactly after every optimizer step;
- only the new speaker row is trainable;
- the discriminator is not allocated on the GPU;
- CUDA forward/backward uses AMP FP16;
- frequent saves write a tiny embedding delta, and the full model is written
  only at the end.

This prevents catastrophic forgetting by construction. It works best when the
new voice is reasonably close to the five-voice manifold learned by the base
model. More aggressive adaptation would require replay/distillation data to
protect the old speakers.

On Colab, install the training dependencies and eSpeak first:

```bash
apt-get update -qq && apt-get install -y espeak-ng
pip install -e ".[train]"
```

### Dataset

Use 22,050 Hz mono WAV files and create `train.txt`:

```text
/content/voice/wavs/0001.wav|5|pt-br|Primeira frase transcrita.
/content/voice/wavs/0002.wav|5|pt-br|Segunda frase transcrita.
```

Every row must use the newly appended speaker ID. Mixed IDs are rejected.

### Append the speaker

The base checkpoint may be FP16 or FP32; its dtype is preserved.

```bash
python examples/finetune/add_speaker.py \
  --checkpoint model/latest_model_fp16.pth \
  --config model/model_config.json \
  --base-speaker frieren \
  --new-speaker marcelo \
  --output-dir /content/marcelo
```

### Adapt on a Colab T4

```bash
python examples/finetune/finetune.py \
  --checkpoint /content/marcelo/checkpoint_surgery.pth \
  --config /content/marcelo/config.json \
  --dataset-dir /content/marcelo/dataset \
  --output-dir /content/marcelo/output \
  --speaker marcelo \
  --precision fp16 \
  --save-dtype fp16 \
  --batch-size 8 \
  --max-steps 2000
```

If the T4 runs out of memory, lower `--batch-size` to 4 or 2. Increasing it
usually improves throughput but does not change the number of trainable values.

Outputs:

- `latest_model_fp16.pth`: complete inference checkpoint;
- `speaker_embedding_delta.pth`: portable new-row delta;
- `model_config.json`: architecture and updated speaker map;
- `train_state.json`: last adaptation step and losses.

If a Colab runtime disconnects after an interval save, restart from the surgery
checkpoint and add:

```bash
--resume-delta /content/marcelo/output/speaker_embedding_delta.pth
```

The FP16 output preserves frozen weights exactly when the input checkpoint is
also FP16. Use `--save-dtype fp32` when adapting from an FP32 base and exact
FP32 preservation is required.

</details>
