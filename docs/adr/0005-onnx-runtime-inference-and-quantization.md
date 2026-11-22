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

## Results (T12, real export)

Command: `uv run stc export --run artifacts/final/seed-42 --out artifacts/bundle --quantize all --reports reports/final` on the owner's RTX 4060 host. The fp32 graph is 601,760,862 bytes (574 MiB), opset 17, dynamic batch and sequence axes. `onnx.checker` and `stc bundle verify` pass.

**fp32 parity (dev, 2,033 utterances).** `max |Δlogit| = 3.92e-3` (limit 5e-3), argmax agreement `1.0`, `passed: true`. The dev sanity run reproduced the selected seed's torch metrics exactly: accuracy 0.8795, macro-F1 0.8699.

**`mlp_only` patterns.** The first real export confirms the node names assumed by `quantization.py`: attention MatMuls live under `/inner/model/layers.N/attn/...` (for example `/inner/model/layers.0/attn/Wqkv/MatMul`) and the classifier head under `/inner/head/dense/MatMul`; `/attn/`, `/head/` and `/classifier/` therefore match real nodes and exclude the intended MatMuls. No pattern correction was needed.

**Quantisation matrix (dev).** Gate: macro-F1 drop ≤ 0.5 pt and label agreement ≥ 99 %. All four variants fail; fp32 ships.

| Variant | Accuracy | Macro-F1 | Agreement | Size (bytes) | Gate |
|---|---|---|---|---|---|
| fp32 | 0.8795 | 0.8699 | 1.0000 | 601,760,862 | reference |
| `int8_dynamic` | 0.8490 | 0.8223 | 0.9139 | 151,317,198 | fail |
| `int8_dynamic_per_channel` | 0.8593 | 0.8318 | 0.9179 | 151,997,624 | fail |
| `int8_dynamic_mlp_only` | 0.8603 | 0.8430 | 0.9228 | 309,082,247 | fail |
| `int8_static_qdq` | 0.0246 | 0.0083 | 0.0251 | 152,669,567 | fail |

The dynamic results are in the same direction as the planning probe but slightly worse: ModernBERT attention is sensitive to int8 weight rounding, and the mlp-only variant pays for a bigger file while still losing 2.7 pt macro-F1. The static QDQ export failed in this run (2.5 % label agreement with fp32, dev macro-F1 0.0083), so it is not a candidate even before the quality gate. The cause was not investigated. The export used the MinMax calibration described below, so this measurement does not show that the architecture cannot be quantized statically.

**Static calibration deviation.** D12 selected percentile calibration for `int8_static_qdq`. Per-channel percentile histograms over this 150 M-parameter encoder peak at 19.1 GiB RSS (measured with ONNX Runtime 1.32.0); the shared 31 GiB host repeatedly OOM-killed the process at ~6.0 GiB anonymous RSS under concurrent portfolio jobs. The variant therefore uses `CalibrationMethod.MinMax` with the same QDQ graph, per-channel weights and 512 fixed-width calibration utterances. This is the only deviation from D12 and it affects a candidate that fails the gate by a wide margin; it does not change the shipped format. `tests/unit/adapters/test_export_torch.py::test_static_variant_uses_minmax_calibration` pins the choice.

**CPU benchmark (S12, `reports/final/benchmark.json`).** Intel Core i5-12400T, ONNX Runtime 1.32.0, fp32, 200 single-utterance runs: 1 thread p50 30.32 ms / p95 43.93 ms; 4 threads p50 15.80 ms / p95 19.90 ms; batch-32 throughput 28.9 / 93.2 utterances/s. The shipped manifest records `quantization.mode = "fp32"`.
