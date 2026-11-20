# Evaluation protocol and report guide

This document describes how the project measures the two classifiers and how to
read every committed report. The decisions behind it are binding: D2 (metrics),
D3 (leakage policy), D4 (split roles and the test lock), D10 (calibration and
abstention), D11 (perturbations), D19 (comparison verdict) and ADR-0004.

## Split roles and the test lock

The official MASSIVE partitions are preserved. Only train is cleaned; dev and
test are reported exactly as released.

| Split | Rows | Role |
|---|---:|---|
| train | 10,898 after cleaning (11,514 raw) | Fitting the baseline and the transformer. |
| dev | 2,033 | Grid search, backbone screening, Optuna selection, seed selection, temperature `T`, review threshold `τ`, quantisation gate. |
| test | 2,974 | One audited final comparison per release candidate. |

`DevOnlyRepository` wraps the dataset repository for every non-final use case,
and `require_test_grant` raises `TestSplitLocked` (exit code 6) when the test
split is requested without a `TestAccessGrant`. There are exactly two audited
readers: the `FinalReport` use case (`stc report final`) and
`stc evaluate --split test --final`. Each appends one JSON line to
`reports/test_access_log.jsonl` with the UTC timestamp, command, git sha, model
sha256 and example count. Log lines are never deleted.

Cleaning follows D3 (`normalize_for_dedup` then three train-only stages), which
drops 616 rows: 356 overlap-with-eval, 43 label-conflict duplicates and 217
exact duplicates. `reports/data_audit.json` records every count and the dropped
ids by reason; `docs/DATA.md` explains the policy.

## Metric definitions

All metrics are computed with the 60-intent label space in lexical order. The
evaluated class set is the labels with at least one true example in the split
(`present_labels`): 59 intents on test (`cooking_query` is absent) and 59 on dev
(`audio_volume_other` is absent). Undefined divisions are 0.

- **Accuracy** — `(1/N) · Σ_i 1[y_i = ŷ_i]`.
- **Per-class precision / recall / F1** — for class `c`, with
  `TP_c = CM[c, c]`, `FP_c = Σ_r CM[r, c] - TP_c` and
  `FN_c = Σ_p CM[c, p] - TP_c`:
  `P_c = TP_c / (TP_c + FP_c)`, `R_c = TP_c / (TP_c + FN_c)`,
  `F1_c = 2·P_c·R_c / (P_c + R_c)`.
- **Macro-F1** — the unweighted mean of `F1_c` over the present labels only.
  This is the primary selection metric (D2).
- **Weighted-F1** — `Σ_c support_c · F1_c / Σ_c support_c`.
- **Top-3 accuracy** — the fraction of rows whose true intent is among the three
  highest scores (`argsort(..., kind="stable")`).
- **Scenario accuracy** — map both the true and the predicted intent to the
  scenario (the intent prefix before the first `_`) and compare the scenarios.

### Confidence intervals and significance

- **Percentile bootstrap**, `B = 2000`, `numpy.random.default_rng(13)`, 95 %
  interval (`2.5` and `97.5` percentiles). The reported interval is clamped to
  include its own point estimate, because a discrete metric on a small resample
  can otherwise produce an interval that contradicts the estimate.
- **Paired Δ bootstrap** — model A and model B are evaluated on the *same*
  resampled row indices, so the interval describes the paired difference
  `metric(A) - metric(B)` rather than two independent intervals.
- **Exact McNemar** — with `b` rows correct only for model A and `c` correct only
  for model B, `n = b + c` and
  `p = min(1, 2 · Σ_{k=0}^{min(b,c)} C(n, k) / 2^n)`.
  The implementation uses integer `math.comb`, not a normal approximation.
- **Verdict (D19)** — `transformer_better` only when the paired Δmacro-F1
  interval excludes zero and McNemar `p < 0.05` favours the transformer; the
  symmetric rule defines `baseline_better`; otherwise `tie`. The README sentence
  is generated from `reports/summary.json`, so it cannot overclaim.

## Calibration and abstention

For logits `z` and a single temperature `T`, the calibrated distribution is
`softmax(z / T)`. `T` is fitted on dev logits by minimising NLL with an
80-iteration golden-section search over `log T ∈ [log 0.05, log 20]`; the
baseline gets its own `T` fitted on its `decision_function` scores.

- **NLL** — `-mean(log p_y)`.
- **Brier** — `mean(Σ_c (p_c - 1[c = y])²)`.
- **ECE** — 15 equal-width bins on the maximum calibrated probability:
  `ECE = Σ_m (n_m / N) · |accuracy_m - confidence_m|`, where `n_m` is the number
  of rows in bin `m`, `confidence_m` is their mean maximum probability and
  `accuracy_m` is their empirical accuracy.
- **Review threshold `τ`** — the smallest calibrated max-probability cut-off at
  which dev selective accuracy is at least 0.95 and the cut-off covers at least
  20 % of dev. If no cut-off qualifies, `τ = 1.0` and the report states that
  abstention has no useful operating point.
- **Coverage / selective accuracy** — at `τ`, coverage is the fraction of rows
  with `confidence >= τ`; selective accuracy is accuracy on that covered subset.
- **`needs_review`** — `confidence < τ`, computed once by
  `CalibratedClassifier` and exposed by the CLI, API and UI. Temperature scaling
  never changes the argmax; only confidence and the review decision move.

## Robustness perturbations

`src/stc/domain/perturbations.py` implements seven pure perturbations. The
evaluated subset is the set of utterances that the function actually changed.

| Name | Definition |
|---|---|
| `no_accents` | NFD, drop combining marks except the tilde on `ñ`/`Ñ`, NFC. |
| `uppercase` | Upper-case the whole utterance. |
| `typos` | For alphabetic words of length ≥ 4, with probability 0.06 apply one deterministic edit (character substitution, deletion or duplication) using `random.Random(f"{seed}:{text}")` with seed 13. The older adjacent-swap idea was replaced by substitution so every edit has Levenshtein distance 1, which the property tests pin. |
| `chat_abbrev` | Whole-word chat abbreviations from `src/stc/domain/lexicon.py`. |
| `regionalize:mx` | Synthetic Mexican Spanish lexicon substitutions. |
| `regionalize:ec` | Synthetic Ecuadorian Spanish lexicon substitutions. |
| `regionalize:ar` | Synthetic Rioplatense lexicon and voseo substitutions. |

For each perturbation and each model the report gives the number of affected
utterances, accuracy before and after on that subset, the paired-bootstrap Δ
accuracy with its 95 % CI, and mean calibrated confidence before/after.
Macro-F1 is deliberately **not** reported on perturbation subsets: they are too
small for stable per-class metrics, and the report says so in its `note` field.

The regionalisation tables are **synthetic**. MASSIVE has no Latin-American
Spanish locale. These probes measure lexical brittleness; they are not evidence
of real LatAm accuracy. The disclaimer is repeated in ADR-0009, the model card
and the README.

## Natural slices

`src/stc/domain/slices.py` produces the following masks on test:

- scenario (18 MASSIVE scenarios);
- length bucket `le4`, `5_8`, `9_12`, `ge13` by whitespace word count;
- `digit:yes` / `digit:no` (at least one decimal digit);
- `accent:yes` / `accent:no` (a diacritic other than the tilde on `ñ`);
- translator `worker:<id>` for workers with at least 30 test rows.

Each slice reports `n`, accuracy with a 95 % CI and, only when `n >= 200`,
macro-F1 with a 95 % CI (`min_macro_f1_support` in the report). The worker
slice also exposes the min/max/spread so the reader can judge whether a score is
driven by one annotator.

## Report files

All reports are deterministic: sorted keys, four-decimal floats, UTF-8 and `\n`.

| File | What it contains | Produced by |
|---|---|---|
| `reports/data_audit.json` | Raw/processed counts, every overlap and duplicate count, drop counts and dropped ids by reason. | `stc data audit` |
| `reports/data_audit.md` | Human-readable audit summary. | `stc data audit` |
| `reports/baseline/search.json` | Every D5 grid configuration with dev accuracy, macro-F1, fit seconds and feature count, plus the winner. | `stc train baseline` |
| `reports/baseline/dev_metrics.json` | Winner's dev metrics, per-class rows, confusion matrix and errors. | `stc train baseline` |
| `reports/experiments/environment.json` | GPU, driver, CPU, RAM, library versions, `uv.lock` sha256, git sha and wall-clock seconds per stage. | `stc doctor --gpu --json` + collector |
| `reports/experiments/screening.json` | Six backbone × seed runs with dev metrics, trainable params, epoch seconds, peak VRAM and CPU latency. | `stc train screen` + collector |
| `reports/experiments/selection.json` | Per-backbone means, the D6 rule and the selected branch/backbone. | `stc train screen` + collector |
| `reports/experiments/full_ft_reference.json` | Full fine-tuning grid by learning rate. | `stc train full-ft` + collector |
| `reports/experiments/search_trials.json` | Every Optuna trial, its state and objective, plus the best trial/config. | `stc train search` + collector |
| `reports/experiments/final_seeds.json` | Three final seeds, selected seed/run, dev mean and std. | `stc train lora` + collector |
| `reports/experiments/search_summary.md` | Search narrative and top trials. | collector |
| `reports/final/metrics.json` | Test accuracy, macro/weighted-F1, top-3, scenario accuracy, CIs, calibration and confusion matrix for both models. | `stc report final` |
| `reports/final/comparison.json` | Paired Δ accuracy/macro-F1 with CIs, McNemar `b`/`c`/`p`, verdict and EN/ES sentences. | `stc report final` |
| `reports/final/per_class.csv` | One row per model × intent: precision, recall, F1 and support (support 0 is flagged by the row itself). | `stc report final` |
| `reports/final/confusion_matrix.csv` | Full 60 × 60 counts per model, long format. | `stc report final` |
| `reports/final/calibration_dev.json` | Dev `T`, `τ`, ECE/NLL before/after and dev coverage/selective accuracy. | `stc calibrate` |
| `reports/final/calibration_test.json` | The same calibration applied to test, including Brier and coverage. | `stc report final` |
| `reports/final/robustness.json` | Perturbation table per model (affected count, before/after, Δ CI, confidence). | `stc report final` |
| `reports/final/slices.json` | Natural-slice table per model. | `stc report final` |
| `reports/final/errors.json` | One entry per test error, with text, true/predicted intent, calibrated confidence and both models' predictions. | `stc report final` |
| `reports/final/export_parity.json` | fp32 ONNX vs torch: max abs logit diff, agreement and pass/fail against the 5e-3 gate. | `stc export` |
| `reports/final/quantization_dev.json` | Four int8 variants, dev accuracy/macro-F1, label agreement, size, gate result and `shipped_format`. | `stc export --quantize all` |
| `reports/final/benchmark.json` | CPU model, ORT version, p50/p95/mean latency per thread count and batch-32 throughput. | `stc benchmark` |
| `reports/figures` | Reliability, confusion, per-class F1, robustness and search figures (PNG). | final report, collector, export |
| `reports/summary.json` | Compact numbers consumed by the README and model-card renderers. | `scripts/render_results.py` |
| `reports/results_table.md` | Markdown results table generated from the summary. | `scripts/render_results.py` |
| `reports/test_access_log.jsonl` | Append-only audit trail: one line per authorised test read. | audited readers |
| `docs/ERROR_ANALYSIS.md` | Qualitative error categories with real ids checked by `scripts/check_error_analysis.py`. | manual, from `errors.json` |

## Re-running the reports

The data and model commands assume `STC_DATA_DIR` points at the prepared local
data and `artifacts/` contains the fitted baseline and final seed. The heavy GPU
stages are documented with measured durations in `docs/TRAINING.md`.

```bash
uv run stc data fetch
uv run stc data verify
uv run stc data prepare
uv run stc data audit --out reports
uv run stc train baseline --out artifacts/baseline --reports reports/baseline
uv run stc export --run artifacts/final/seed-42 --out artifacts/bundle --quantize all --reports reports/final
uv run stc calibrate --bundle artifacts/bundle --baseline artifacts/baseline --reports reports/final
uv run stc benchmark --bundle artifacts/bundle --threads 1,4 --runs 200 --out reports/final/benchmark.json
uv run stc report final --bundle artifacts/bundle --baseline artifacts/baseline --out reports/final --figures reports/figures
uv run python scripts/collect_experiment_reports.py --artifacts artifacts --out reports/experiments --figures reports/figures
uv run python scripts/render_results.py --summary reports/summary.json --table reports/results_table.md
uv run python scripts/render_model_card.py --template docs/model_card.template.md --summary reports/summary.json --out MODEL_CARD.md
uv run python scripts/validate_reports.py all
```

The heavy training commands (screening, full fine-tuning, Optuna search and the
three final seeds) are in `docs/TRAINING.md`; their compact results are produced
by `scripts/collect_experiment_reports.py`.

## Re-runs of the final evaluation

The audit log currently contains **one** line, so there has been one audited
test read for v0.1.0 and no re-runs to explain:

| Timestamp (UTC) | Command | git sha | Model sha256 | Examples |
|---|---|---|---|---:|
| 2026-11-20T17:02:19 | `stc report final --bundle artifacts/bundle --baseline artifacts/baseline --dataset massive` | `c6523f4` | `8a438c1c…75fbd5` | 2,974 |

If a future bug fix forces a second read, append a new row here with the reason;
never delete or rewrite an existing log line.

## Reproducibility notes

- Python, NumPy and torch seeding goes through `set_all_seeds(13)`; torch uses
  `use_deterministic_algorithms(True, warn_only=True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Bit-exact reproduction across GPUs and
  drivers is not claimed.
- Bootstrap and paired-bootstrap draws use `default_rng(13)` with a fixed
  `B = 2000`; McNemar is exact.
- The Baseline uses a fixed grid (`configs/baseline.toml`) and a fixed
  tie-break: macro-F1, then accuracy, then fewer features.
- Selection uses dev only. The test numbers in `reports/final` were produced once
  by the command in `reports/test_access_log.jsonl`.
- Tests never use the network. The fixtures under `src/stc/_fixture` and
  `tests/fixtures` are deterministic and committed.
- The reports record the git sha and `uv.lock` sha256 of the run in
  `reports/experiments/environment.json`; the released model's manifest records
  the training run, tokenizer length and quantisation mode.

## Known limitations

- Only `es-ES` is evaluated; the LatAm probes are synthetic (see above).
- Test has no `cooking_query` example, so macro-F1 is over 59 of 60 intents.
- Perturbation subsets report accuracy, not macro-F1, by design.
- Temperature scaling fixes confidence, not ranking; high-confidence label-noise
  errors remain visible in `docs/ERROR_ANALYSIS.md`.
- CPU latency depends on the host; `reports/final/benchmark.json` records the CPU
  model and ONNX Runtime version next to the numbers.
