# ADR-0003 — Backbone and adaptation method

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

The classifier needs an encoder that is strong in Spanish, small enough to run
on CPU after ONNX export, and licensed so that fine-tuned weights can be
redistributed. It also needs an adaptation method that fits an 8 GB RTX 4060,
can be reproduced with an explicit loop, and can later be merged into a single
ONNX graph. D6 pre-registers the candidates and the selection rule; D7 fixes
LoRA hyper-parameters and the custom training loop.

## Candidate table

| Backbone | Architecture | Parameters | Licence | Role |
|---|---|---|---|---|
| `BSC-LT/MrBERT-es` | ModernBERT encoder | 150,295,040 | Apache-2.0 | Primary |
| `bertin-project/bertin-roberta-base-spanish` | RoBERTa encoder | 124,695,640 | CC BY 4.0 | Screening fallback |
| `Qwen/Qwen3-0.6B` | decoder | 751,632,384 | Apache-2.0 | Reference only, never shipped |

BETO was rejected because its licence metadata is missing and its README warns
that training-corpus licences may be incompatible. `roberta-base-bne` was
rejected because its weights are unavailable (empty repository, `config.json`
404 on 2026-11-19). `xlm-roberta-base` was rejected because it doubles the
artifact size without quality evidence. `docs/DATA.md` records the full list.

## Decision

1. **Mechanical rule.** Among exportable candidates (MrBERT-es and bertin),
   choose the highest mean dev macro-F1. MrBERT-es wins when it is within 1.0
   percentage point of bertin; bertin wins only when it is more than 1.0 point
   ahead. Qwen3-0.6B is never selectable (`exportable = false`) because it is a
   decoder reference, not a deployable encoder. The rule lives in
   `application/services/backbone_selection.py`; tests pin all three branches.
2. **LoRA** (`peft.LoraConfig(task_type=SEQ_CLS, target_modules="all-linear")`)
   is the adaptation method. It keeps trainable parameters under 25 % (measured
   2.24 % on MrBERT-es at r=16), fits comfortably in 8 GB at batch 32, and
   merges into the base weights before ONNX export.
3. **Custom PyTorch loop, not HF `Trainer`.** Tokenisation, batching, seeding,
   gradient clipping, logging and dev evaluation are explicit and shared with
   inference. This keeps calibration/analysis inputs identical to the trained
   artifact and avoids Trainer-specific logging formats.

## Alternatives

- Full fine-tuning for the shipped model: measured peak 2.90 GiB versus
  1.52 GiB for LoRA, and no evidence of better dev macro-F1 in the probe.
- Frozen encoder + linear head: cheap, but the probe's 4-epoch LoRA run already
  matched a tuned TF-IDF baseline, so freezing the encoder would give up the
  main comparison.
- Distillation or a smaller multilingual model: out of scope for v0.1.0.

## Consequences

- The selected checkpoint must be exportable to ONNX with eager attention;
  decoders are excluded by construction.
- The selection rule is data-driven but pre-registered: nobody can swap
  backbones after seeing test numbers.
- LoRA adapters are shipped as a separate small release asset, while the merged
  fp32 ONNX graph is the runtime artifact.

## Evidence

- `uv run pytest tests/unit/application/test_backbone_selection.py -q`
- `uv run pytest -q -m torch` (tiny-backbone trainer, search and screening)
- Probe measurements in the plan's “Local measurements” table (2026-11-19).

## Results

Measured on 2026-11-20 on the owner's RTX 4060 8 GB (`reports/experiments/`,
commands in `docs/TRAINING.md`; every number below comes from those committed
reports or from the run metadata they cite). Training used the cleaned MASSIVE
es-ES train split (10,898 rows) and selected on dev (2,033 rows) only.

### Backbone screening (D7 recipe, two seeds each)

| Backbone | Exportable | Dev macro-F1 (mean ± std) | Dev accuracy (mean) | Trainable params | Mean epoch (s) | Peak VRAM (MiB) | CPU p50 (ms)* |
|---|---|---|---|---|---|---|---|
| `BSC-LT/MrBERT-es` | yes | 0.8313 ± 0.0006 (0.8309, 0.8317) | 0.8667 | 3,496,056 | 31.9 | 1,666 | 116.4 |
| `bertin-project/bertin-roberta-base-spanish` | yes | 0.7997 ± 0.0031 (0.8019, 0.7975) | 0.8463 | 3,927,672 | 21.5 | 1,666 | 82.9 |
| `Qwen/Qwen3-0.6B` | no (decoder) | 0.8418 ± 0.0049 (0.8453, 0.8383) | 0.8579 | 10,153,984 | 193.2 | 6,707 | 224.9 |

\* Single-utterance CPU latency measured *while the GPU job was running*, with
the CPU otherwise busy; it is useful only for relative ordering in this table.
T12 replaces it with the committed `stc benchmark` numbers. Qwen3-0.6B hit CUDA
OOM at batch 32 and, as D7 allows, retried at batch 16 with gradient
accumulation 2 (`oom_retry: true`); it remains a screening reference and is
never exportable.

**Selection:** `BSC-LT/MrBERT-es`, branch `mrbert_ahead`. MrBERT-es is 3.16
points of dev macro-F1 ahead of bertin (0.8313 vs 0.7997), far outside the 1.0
point margin, so the D6 fallback rule did not trigger. Qwen3-0.6B had the
highest screening mean (0.8418) but is excluded by construction because it is a
decoder; its per-epoch cost was also ~6× MrBERT-es.

### Full fine-tuning reference (MrBERT-es, seed 13, 4 epochs, batch 32)

| Learning rate | Dev accuracy | Dev macro-F1 | Peak VRAM (MiB) | Wall (s) |
|---|---|---|---|---|
| 2e-5 | 0.8716 | 0.8428 | 2,961 | 176.9 |
| 3e-5 | 0.8682 | 0.8371 | 2,964 | 166.3 |
| 5e-5 | 0.8795 | **0.8645** | 2,964 | 163.3 |

Full fine-tuning trains 150,289,980 parameters and peaks at ~2.96 GiB. Its best
single run (0.8645) is 0.09 points above the LoRA three-seed mean (0.8636) and
0.54 points below the best LoRA seed (0.8699): no evidence that updating every
weight is worth ~43× the trainable parameters or ~2.3× the VRAM.

### Optuna search (24 trials, TPE, seed 13)

The search completed 24/24 trials (8 COMPLETE, 16 PRUNED) with best dev
macro-F1 **0.8627** (trial 5). Best configuration:

`r=32`, `alpha=32`, `lora_dropout=0.0`, `lr=3.632e-4`,
`weight_decay=0.01`, `epochs=6`, `batch_size=16`, `warmup_ratio=0.06`,
`class_weighting=none`.

The 16 pruned trials were the slowest/highest-epoch configurations that fell
below the running median after the first epoch; the sampler itself is the D7
pre-registered TPE, and the best value should be read with the ~±0.005 seed
noise visible below. The search objective ranks configurations, not the final
model: the three-seed rerun of the winning configuration is the number used for
selection.

### Final LoRA seeds (best configuration)

| Seed | Dev accuracy | Dev macro-F1 | Dev NLL | Wall (s) | Peak VRAM (MiB) |
|---|---|---|---|---|---|
| 13 | 0.8741 | 0.8627 | 0.7857 | 239.4 | 1,261 |
| 42 | 0.8795 | **0.8699** | 0.7695 | 249.0 | 1,261 |
| 2026 | 0.8746 | 0.8583 | 0.7673 | 266.1 | 1,261 |

Shipped seed = **42** (highest dev macro-F1; tie-breaker not needed). Three-seed
dev mean = **0.8636**, sample std = **0.0059**. LoRA r=32 trains 6,899,832 of
157,143,672 parameters (4.39 %) at 1.26 GiB peak.

### Honest interpretation

On dev, the tuned LoRA model is 2.07 points of macro-F1 above the committed
TF-IDF + linear baseline (0.8636 vs 0.8429) and 2.8 points above its accuracy
(0.8795 vs 0.8495). The probe's much smaller gap did not reproduce with the
searched configuration, but dev is used for selection and cannot settle the
question: T12 reports the single audited test comparison, bootstrap intervals
and McNemar verdict. The full-FT reference and the search show the usual
variance of a 60-class problem with few examples in the tail; any test
difference smaller than ~1 point should be treated as unresolved unless the
paired intervals and McNemar agree. No test number is claimed here.

Environment and wall-clock seconds per stage are committed in
`reports/experiments/environment.json` (screening 1,972 s, full-FT 507 s,
search 1,316 s, final seeds 754 s; git sha `d5cb1f3`, uv.lock sha256 recorded
there).
