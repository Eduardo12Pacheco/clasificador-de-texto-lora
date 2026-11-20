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

Filled by T11 with the real screening table, full-FT comparison and final
three-seed dev mean ± std. No numbers are claimed here until that run exists.
