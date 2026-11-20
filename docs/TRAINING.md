# Training guide

This guide reproduces the committed training reports on a machine with an
8 GB NVIDIA GPU. The runtime path does not need any of this: inference is CPU-only
ONNX Runtime. Training lives in the `train` extra and in
`src/stc/adapters/training`, the only package that imports torch, transformers,
PEFT and Optuna.

## Hardware and software prerequisites

- NVIDIA GPU with at least 8 GB of VRAM (the reference machine is an RTX 4060
  with 7,848.6 MiB usable), a current CUDA driver, and a CUDA-enabled torch
  wheel.
- Python 3.12 managed by `uv`, plus git.
- CPU RAM: 16 GiB is enough for the LoRA runs; the measured host had 31.08 GiB.
- The pinned MASSIVE archive (39,500,415 bytes, sha256
  `7df623fd…153f8`) is downloaded once by `stc data fetch`.

The reference environment recorded in `reports/experiments/environment.json`:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, driver 620.70.06, CUDA 13.2, bf16 supported |
| CPU | 12th Gen Intel Core i5-12400T, 12 threads |
| RAM | 31.08 GiB |
| Python | 3.12.13 (uv) |
| torch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| peft | 0.22.0 |
| optuna | 5.1.0 |
| mlflow | 3.18.0 |
| ONNX Runtime | 1.32.0 |

## Setup

```bash
uv sync --locked --extra train --group dev
uv run stc doctor --gpu --json
```

`stc doctor --gpu` must report `"cuda_available": true` and a device name. If it
does not, the CUDA wheel is missing or the driver is not visible; do not train on
CPU and expect the committed timings.

The dataset directory is intentionally git-ignored. Point `STC_DATA_DIR` at a
local directory before the first prepare:

```bash
export STC_DATA_DIR="$PWD/data"
uv run stc data fetch
uv run stc data prepare
```

`stc data prepare` must print `train=10898 dev=2033 test=2974 dropped=616`.
`artifacts/baseline` must already exist from the baseline task (or be rebuilt
with `uv run stc train baseline --out artifacts/baseline --reports reports/baseline`).

On a machine shared with other GPU jobs, wrap each long command with the shared
lock (the commands below omit it for readability):

```bash
flock /tmp/portfolio-gpu.lock uv run stc train screen --config configs/screening.toml
```

## Capture the environment

```bash
mkdir -p artifacts reports/experiments reports/figures logs
uv run stc doctor --gpu --json > artifacts/environment.json
```

The collector later adds `stages_seconds` by summing each run's `run.json`.

## Step 1 — backbone screening

The D7 screening recipe trains three backbones × two seeds with r=16, alpha=32,
dropout 0.05, weight decay 0.01, warm-up 0.06, batch 32 and four epochs. The
learning rate is 3e-4 for the encoders and 1e-4 for the decoder reference.

```bash
uv run stc train screen --config configs/screening.toml --out artifacts/screening --tracker jsonl,mlflow
```

Measured duration: **1972.199 s** (about 32 min 52 s). Outputs
`artifacts/screening/screening.json` and `artifacts/screening/selection.json`.
The pre-registered D6 rule runs in `src/stc/application/services/backbone_selection.py`.
The committed run selected `BSC-LT/MrBERT-es` with branch `mrbert_ahead`.

If a CUDA out-of-memory error occurs, the trainer retries the run once with
`batch_size=16` and gradient accumulation 2, then records `oom_retry: true`.
Qwen3-0.6B needed that retry; it remains a non-exportable reference.

## Step 2 — full fine-tuning reference

This is the comparison point, not the shipped model. It updates all weights and
uses the same trainer with the adapter disabled.

```bash
uv run stc train full-ft --backbone BSC-LT/MrBERT-es --lrs 2e-5,3e-5,5e-5 --epochs 4 --out artifacts/full_ft --tracker jsonl,mlflow
```

Measured duration: **506.509 s** (about 8 min 27 s). Peak VRAM: 2,964 MiB. The
best reference run (lr 5e-5) reached dev macro-F1 0.8645, which is why the LoRA
recipe is preferred: it trains ~4.4 % of the parameters at 1,261 MiB for the
final runs.

## Step 3 — Optuna hyper-parameter search

The D7 space is pre-registered in `configs/lora_search.toml`: 24 TPE trials with
`seed=13`, `multivariate=True` and `MedianPruner(n_startup_trials=6,
n_warmup_steps=1)`. Trials prune on per-epoch dev macro-F1; the objective is dev
macro-F1 after the last epoch.

```bash
uv run stc train search --backbone BSC-LT/MrBERT-es --trials 24 --config configs/lora_search.toml --out artifacts/search --tracker jsonl,mlflow
```

Measured duration: **1315.813 s** (about 21 min 56 s), 24/24 trials (8 COMPLETE,
16 PRUNED). Outputs include `artifacts/search/search_trials.json`,
`artifacts/search/best_config.toml`, history and importance figures. The
committed best trial (trial 5) is r=32, alpha=32, dropout 0.0, lr≈3.63e-4,
weight decay 0.01, 6 epochs, batch 16, warm-up 0.06 and no class weighting.

Copy the generated config into the repository when you want the final runs to
use it:

```bash
cp artifacts/search/best_config.toml configs/best_lora.toml
```

## Step 4 — final seeds

Train the winning configuration with the three pre-registered seeds and select
the shipped seed on dev (highest dev macro-F1; tie → lower dev NLL).

```bash
uv run stc train lora --config configs/best_lora.toml --backbone BSC-LT/MrBERT-es --seeds 13,42,2026 --out artifacts/final --tracker jsonl,mlflow
```

Measured duration: **754.476 s** (about 12 min 34 s). Peak VRAM: 1,261 MiB per
seed. The committed run selected seed 42 (dev macro-F1 0.8699); the three-seed
mean is 0.8636 ± 0.0059. Outputs: `artifacts/final/seed-13`, `seed-42`,
`seed-2026` and `artifacts/final/selected.json`.

## Step 5 — collect and validate the compact reports

Large run directories stay in `artifacts/` (git-ignored). Only summaries are
committed.

```bash
uv run python scripts/collect_experiment_reports.py --artifacts artifacts --out reports/experiments --figures reports/figures
uv run python scripts/validate_reports.py experiments
```

The collector writes `reports/experiments/environment.json`, `screening.json`,
`selection.json`, `full_ft_reference.json`, `search_trials.json`,
`final_seeds.json` and `search_summary.md`, and copies the search figures into
`reports/figures`.

## Monitoring, tracking and resuming

- **JSONL tracker** writes one event per line to `<run>/tracker.jsonl` and keeps
  `run.json`/`metrics.json` next to the checkpoint. This is the default.
- **MLflow** records the same parameters, metrics and artifacts in
  `sqlite:///artifacts/mlflow.db`. Inspect it locally on a port inside the
  project's 19000–19999 range:

  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 19081
  ```

- **Optuna** stores its study in `artifacts/search/optuna.db`. Resume an
  interrupted search with `--resume`; completed trials are not repeated.
- **Screening and final seeds** skip a run whose `run.json` already exists. Rerun
  the same command to continue after a crash or a session end.
- `stc doctor --gpu --json` is the first diagnostic when a run does not start.

## Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| `cuda_available: false` | `nvidia-smi`, `stc doctor --gpu --json` | Install a CUDA torch wheel; do not fall back to CPU training. |
| CUDA out of memory | run log, `peak_vram_mib` | The trainer retries once at batch 16/grad-accum 2; otherwise lower `batch_size` or `max_length` in the TOML. |
| Loss is NaN or dev macro-F1 < 0.78 on screening | `labels.json` order, pad id, `max_length`, learning rate | The D7 sanity floor is 0.78; fix the input pipeline before continuing. |
| Search trial fails for a non-OOM reason | `search_trials.json` state | The search records the trial and continues; more than six failed trials means stop and investigate. |
| MLflow UI port is busy | `STC_API_PORT` range note | Use another port in 19000–19999. |
| Tokenizer does not load | `tokenizer.json` in the backbone cache | D8 uses the Rust `tokenizers` file, not `AutoTokenizer`; the adapter falls back to the decoder end-of-text token when `<pad>` is absent. |

## Swapping the backbone

Pass `--backbone <hf-id>` to `stc train full-ft`, `stc train search` and
`stc train lora`, or edit the candidate list in `configs/screening.toml` and rerun
screening. The backbone must be an encoder that ships `tokenizer.json` and can be
merged and exported to ONNX; decoder-only models are screening references and
are never shipped. A change to the candidate set or the selection rule is a
Decision change (D6/ADR-0003), not a local experiment: record it in the ADR and
re-run selection before touching test.

To change the adapter recipe, edit `configs/best_lora.toml` (the generated file
carries the trial values) and rerun the final seeds. Keep the fixed D7 pieces
(maximum length 64, eager attention, the custom training loop) unless a decision
changes them.

## Retraining on a different dataset

1. Implement `RawDatasetSource` and `DatasetParser` for the new format and wire
   them in `build_data_services` in `src/stc/bootstrap.py`.
2. Keep the train/dev/test roles and run `stc data prepare`; the leakage audit
   and D3 cleaning must pass before training.
3. Rebuild the baseline (`stc train baseline`) and re-run the full D7 pipeline.
   Labels are derived from the processed data in lexical order, so a new label
   space changes `labels.json` and invalidates old bundles.
4. Update `docs/DATA.md` with the source, licence, checksum and lineage, and the
   model card with the new data statement. Changing the dataset affects D1, so
   it needs a Decision update before the results are published.

The test split stays locked: a new dataset gives you a new one-time audited
comparison, and every read still appends to `reports/test_access_log.jsonl`.

## VRAM and runtime budget (measured)

| Stage | Wall time | Peak VRAM | Notes |
|---|---:|---:|---|
| Screening (6 runs) | 1972.199 s | 1,666 MiB (LoRA), 6,707 MiB (Qwen3 retry) | Includes the decoder reference. |
| Full fine-tuning (3 lrs) | 506.509 s | 2,964 MiB | Updates all 150 M parameters. |
| Optuna search (24 trials) | 1315.813 s | 1,666 MiB | Pruning reduces the average. |
| Final seeds (3) | 754.476 s | 1,261 MiB | Winning r=32 configuration. |

The total measured pipeline is about 76 minutes on the reference machine. These
are measured values, not guarantees; the committed reports are the source of
truth.

## CPU-only caveat

Training requires the `train` extra and a GPU. CPU inference is the supported
product path: install the wheel without the extra, pull the ONNX bundle and run
`stc predict` or `stc serve`. The CI `smoke-install` job asserts that
`import torch` fails inside the installed tool environment while inference still
works.
