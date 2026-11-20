# ADR-0006 — Calibration and abstention

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

The classifier must expose a confidence that a user can act on. A softmax
maximum is not a calibrated probability: a freshly fine-tuned transformer is
usually over-confident, and a linear baseline's `decision_function` margins are
not probabilities at all. The product also needs an abstention policy: when the
model is not sure, the CLI, API and UI must say so instead of guessing.

S8 fixes the deliverables: ECE, NLL and Brier before and after calibration on
test; a review threshold chosen on dev for 0.95 selective accuracy; test
coverage and selective accuracy at that threshold; and `needs_review` surfaced
by every interface. D4 forbids using the test split for any of these choices.

## Decision

1. **Temperature scaling** (`p = softmax(z / T)`) is the calibration method.
   `T` is fitted on dev logits only by minimising NLL with a pure-numpy
   golden-section search over `log T ∈ [log 0.05, log 20]`, 80 iterations (D10).
   The baseline receives its own `T` fitted on its decision-function scores.
2. **ECE uses 15 equal-width bins on the maximum probability** (D10). NLL and
   Brier are always reported with ECE so a single binning artefact cannot hide a
   worse probability distribution.
3. **The review threshold `τ`** is the smallest calibrated max-probability
   cut-off at which dev selective accuracy is at least 0.95 while covering at
   least 20 % of dev. If no cut-off qualifies, `τ = 1.0` and the report states
   that abstention has no useful operating point (`Calibration` with a
   threshold of 1.0 means every prediction is reviewed).
4. **`needs_review` is `confidence < τ`**, computed by the
   `CalibratedClassifier` decorator so the CLI, API and UI share one rule.
5. **Argmax never changes under temperature scaling**; only confidence and the
   abstention decision do. A test asserts this invariant.

## Alternatives

- **Platt scaling** (a logistic fit per class or on the top score): more
  parameters and a second model to fit, with no expected gain for a single
  monotone distortion of an already-trained logit vector.
- **Isotonic regression**: flexible but piecewise-constant, prone to
  overfitting with the dev size (2,033 rows) and unable to extrapolate beyond
  the observed confidence range.
- **Calibration ensembles / deep ensembles**: require retraining the backbone
  several times, which conflicts with the measured budget and adds GPU hours
  without improving the abstention story.
- **Fixed threshold (e.g. 0.9)**: not reproducible across models and datasets;
  D10's rule is derived from dev evidence and reported with its coverage.
- **Fitting `T` on test**: forbidden by D4 and would make every test metric
  optimistic.

## Consequences

- Every interface can show a calibrated confidence and a review flag; the
  model card and README can quote ECE before/after and the threshold's test
  coverage. This is the evidence that the "confidence" number is meaningful.
- Temperature scaling is a single monotone parameter: it cannot fix a model
  whose ranking is wrong, and it does not change accuracy. The reports therefore
  keep accuracy, macro-F1, NLL and Brier next to ECE.
- ECE depends on the binning rule. The 15-bin choice is fixed in code and
  documented, and ECE is never the sole calibration metric.
- The baseline's `decision_function` is treated as a logit, so the baseline's
  calibration is comparable with the transformer's on the same rule.
- `stc calibrate` (T8) will write `calibration.json` with `temperature` and
  `review_threshold` next to the bundle; T12 fills the results addendum.

## Evidence

- `uv run pytest tests/unit/domain/test_calibration.py -q` — temperature
  recovery within 5 % on synthetic logits, ECE bounds, threshold rule and
  selective metrics.
- `uv run pytest tests/unit/application/test_evaluate_model.py -q` — the
  evaluate use case reports uncalibrated and calibrated ECE/NLL/Brier and the
  coverage at `τ`.
- `src/stc/domain/calibration.py` — `fit_temperature`,
  `choose_review_threshold`, `selective_metrics`, `Calibration`.
- `src/stc/application/services/calibrated_classifier.py` — the shared
  `predict`/`predict_proba`/`needs_review` implementation.
- `docs/adr/0004-evaluation-protocol-and-leakage-policy.md` — dev-only model
  selection, which this ADR follows.

## Results (T12, real models)

Fitted on dev (2,033 utterances) with the D10 rule, then applied unchanged to the audited test split. `reports/final/calibration_dev.json` and `reports/final/calibration_test.json` hold the numbers.

| Model | T | τ | Dev ECE before → after | Dev NLL before → after | Test ECE before → after | Test coverage at τ | Test selective accuracy |
|---|---|---|---|---|---|---|---|
| TF-IDF baseline | 0.7454 | 0.7195 | 0.0832 → 0.0265 | 0.6366 → 0.5842 | 0.0849 → 0.0253 | 0.7690 | 0.9353 |
| LoRA transformer | 2.2872 | 0.7655 | 0.0826 → 0.0169 | 0.7695 → 0.4814 | 0.0938 → 0.0181 | 0.8083 | 0.9509 |

The transformer is under-confident after fine-tuning (T > 1), the linear baseline is over-confident (T < 1). Temperature scaling cuts the transformer's test ECE by 81 % and its test NLL by 38 %, without changing any argmax. At the dev-chosen τ the transformer covers 80.8 % of test utterances at 95.1 % selective accuracy; the baseline covers 76.9 % at 93.5 %. The `needs_review` flag uses exactly these thresholds through `CalibratedClassifier` in the CLI, API and UI.

One caveat is visible in the error analysis: some high-confidence mistakes (for example `ex-612` and `ex-1639`) look like gold-label noise rather than calibration failure. Temperature scaling cannot repair a wrong ranking; the thresholds only decide when to abstain.
