# ADR-0004 — Evaluation protocol and leakage policy

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

MASSIVE is assembled from translations of SLURP and contains genuine duplicate
utterances across partitions. Reading the test split early, or training on rows
whose text also appears in dev/test, silently inflates the final comparison and
makes the baseline-vs-transformer verdict untrustworthy. The protocol must be
pre-registered, reproducible and visible in committed artifacts.

## Decision

### Split roles (D4)

- **train** — model fitting only.
- **dev** — hyper-parameter search, backbone screening, model selection,
  temperature/fitting, review-threshold choice and the quantisation gate.
- **test** — exactly one audited final evaluation per release candidate. It is
  read only through a `TestAccessGrant` issued to `stc evaluate --split test
  --final` and `stc report final`; every read appends one line to
  `reports/test_access_log.jsonl`. Every other use case receives a
  `DevOnlyRepository` that raises `TestSplitLocked` (exit 6) for test.

Seed variance is measured on **dev** (three seeds), never on test.

### Cleaning policy (D3)

`normalize_for_dedup` is `casefold → NFKD → drop combining marks → remove
`[^\w\s]` → collapse whitespace → strip`. Train rows are sorted by integer id
and three stages run on train only:

1. drop rows whose normalised text appears in dev or test (`OVERLAP_WITH_EVAL`);
2. drop every row of a group whose normalised text carries more than one intent
   (`LABEL_CONFLICT_DUPLICATE`);
3. keep the lowest-id row of every remaining duplicate group
   (`EXACT_DUPLICATE` for the others).

Dev and test are never modified. Translator `worker_id` cannot be used to
regroup splits: 19 of 20 workers appear in more than one partition.

### Metrics and statistics (D2)

Primary metric: macro-F1 over classes with at least one true example in the
evaluated split, `zero_division=0`. Secondary: accuracy, weighted-F1, top-3
accuracy, scenario accuracy, ECE/NLL/Brier. Confidence intervals: percentile
bootstrap, B = 2000, `numpy.random.default_rng(13)`, paired resampling for the
baseline-vs-model Δ. Significance: exact two-sided McNemar on per-example
correctness.

## Alternatives

- **Re-split the corpus from scratch** — discards the official, human-designed
  partitions and makes results incomparable with published MASSIVE numbers.
- **Deduplicate every split** — would hide the real contamination from the
  numbers; the protocol reports dev/test duplication instead of erasing it.
- **Train on all rows and report test after tuning** — leaks the test split
  into model selection and cannot be audited.
- **Group splits by `worker_id`** — impossible without losing most data because
  workers span partitions.

## Consequences

- Train shrinks from 11,514 to 10,898 rows (616 dropped: 356 overlap, 43
  label-conflict, 217 exact duplicates).
- Dev/test keep 45 overlapping normalised texts and their own duplicate groups;
  those numbers are committed in `reports/data_audit.json` rather than hidden.
- The test evaluation is expensive to repeat by design; re-runs must be recorded
  in the evaluation log (`docs/EVALUATION.md`).
- The final comparison is statistical: a small transformer lead may be a tie.

## Evidence

- `uv run stc data prepare` → `train=10898 dev=2033 test=2974 dropped=616`.
- `uv run stc data audit --out reports` → exact counts in the acceptance list
  (dev∩test 45, dev-in-train 123/7 conflicts, test-in-train 152/6, 264 train
  duplicate groups, 19 label-conflict groups, dev 18 / test 31 groups).
- `tests/unit/domain/test_leakage.py` and
  `tests/unit/application/test_dev_only_repository.py` enforce the policy and
  the lock in the test suite.
