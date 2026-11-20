# ADR-0005 — ONNX Runtime inference and the quantisation gate

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

S11/S12 require a CPU-only inference path that never imports `torch`, an ONNX
export of the merged LoRA model with measured fp32 parity, and an honest
decision about int8 quantisation. The planning probe on the owner's machine
showed that fp32 ONNX is exact and already fast, while default dynamic int8
loses several points of macro-F1 on a ModernBERT architecture. Shipping int8
blind would silently degrade the product; shipping fp32 blindly would ignore a
pure win when a variant passes.

## Decision

1. **ONNX Runtime is the only runtime scorer.** `OnnxClassifier`
   (`adapters/inference/onnx_classifier.py`) loads `model.onnx`, the Rust
   `tokenizers` file and `labels.json`, uses `CPUExecutionProvider`, honours an
   explicit `intra_op_num_threads`, and returns float32 logits in input order.
   The runtime package contract forbids `torch`, `transformers`, `peft`,
   `onnx` and `optuna` in every runtime adapter; only `onnxruntime` and
   `tokenizers` are imported.
2. **Export pipeline.** `OnnxExporter` rebuilds the base backbone with eager
   attention, attaches the PEFT adapter, calls `merge_and_unload()`, wraps the
   model so the graph emits a single `logits` output, and exports with
   `opset_version=17`, `dynamo=False`, dynamic batch and sequence axes. It
   copies the tokenizer, writes `labels.json`, and records a manifest.
3. **fp32 parity is a hard gate.** A merged-model vs ONNX comparison on dev
   passes only when `max |Δlogit| <= 5e-3` and argmax agreement is exactly
   `1.0`. `stc export` writes `export_parity.json` and fails the run if the
   parity check fails.
4. **Quantisation candidate matrix (D12).** Four variants are built and
   evaluated on dev: `int8_dynamic`, `int8_dynamic_per_channel`,
   `int8_dynamic_mlp_only` (exclude attention, head and classifier nodes by
   name) and `int8_static_qdq` (QDQ, per-channel, 512 fixed-width calibration
   utterances, percentile). A variant ships only when its macro-F1 drop versus
   fp32 is `<= 0.5` point **and** argmax agreement is `>= 99 %`. The smallest
   passing variant ships; otherwise fp32 ships. `quantization_dev.json` records
   every variant and the shipped format.
5. **Static calibration fallback.** ONNX Runtime's percentile calibrator cannot
   build a 2048-bin histogram when a tensor has a single value (the constant
   attention mask of the tiny test graph is one such case). The exporter falls
   back to MinMax calibration for that export only; the real MrBERT-es export
   uses percentile as decided.

## Alternatives

- TorchScript inference: keeps `torch` in the runtime, violating S12 and the
  CPU-only install story.
- `optimum` export: extra dependency for a path we already control explicitly.
- Shipping int8 unconditionally: the probe measured a 3–4.5 pt macro-F1 loss.
- Shipping fp32 unconditionally: the negative result must be measured, not
  assumed; a future backbone may pass the gate.

## Consequences

- The installed tool works on a CPU with no training extra; a CI smoke test
  asserts `import torch` fails inside the tool environment.
- The int8 decision is reproducible from committed dev reports; the gate can
  flip without code changes once a run passes.
- `scripts/list_onnx_nodes.py` exists so the first real export can confirm the
  `mlp_only` node-name patterns before they are trusted.
- The first real export (T12) appends measured results to this ADR.

## Evidence

- Planning probe (not committed): fp32 parity 100 % argmax agreement, max
  |Δlogit| 1.39e-3; default int8 0.8234 accuracy / 0.7827 macro-F1 / 88.5 %
  agreement; per-channel 0.8328 / 0.7983 / 90.1 %; fp32 0.8613 / 0.8282.
- `uv run pytest tests/unit/adapters/test_onnx_classifier.py -q`
- `uv run pytest tests/unit/adapters/test_parity.py -q`
- `uv run python -m pytest tests/unit/adapters/test_export_torch.py -q -m torch`
- `uv run python -c "import sys, stc.adapters.inference.onnx_classifier; assert 'torch' not in sys.modules"`
