# ADR-0008 — Experiment tracking

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

T7 runs screening, a full fine-tuning reference, an Optuna search and final
seed runs. The evidence must be reproducible and reviewable without committing
large checkpoints or databases. The project is local-first and must not require
an account or a hosted tracking server. D9 fixes a port and its adapters; D5
requires the Optuna search to be resumable.

## Decision

1. **`ExperimentTracker` port** (`application/ports.py`) with four adapters:
   - `JsonlTracker` (default): one JSONL event stream per run directory, no
     extra dependency and trivially diffable after collection.
   - `MlflowTracker`: local SQLite (`sqlite:///artifacts/mlflow.db`), storing
     params, per-epoch metrics, tags (git sha, backbone, seed) and `run.json`.
   - `NullTracker`: Null Object for commands that run without tracking.
   - `CompositeTracker`: fan-out with per-child error isolation so a broken
     MLflow store cannot lose a training run.
2. **Pure `TrackerFactory`.** It parses `STC_TRACKER` (`jsonl`, `jsonl,mlflow`,
   `null`), rejects unknown names with `InvalidInputError`, and receives builder
   callables from the composition root. The application layer never imports
   adapters, which keeps the hexagonal contracts intact.
3. **Optuna storage is a local SQLite file** (`<out>/optuna.db`) with
   `load_if_exists=True`. `--resume` runs only the missing trials up to the
   requested budget; a second invocation prints `0 new trials`.
4. **Only compact summaries are committed.** `scripts/collect_experiment_reports.py`
   strips absolute paths and large trial payloads, writes deterministic JSON
   under `reports/experiments/`, and copies the search figures. Databases,
   checkpoints and trial directories stay under `artifacts/`, which is ignored.

## Alternatives

- Weights & Biases or a hosted MLflow: requires an account/network and violates
  the zero-account constraint.
- Plain stdout logs: not queryable and easy to lose.
- Committing the SQLite databases: binary, large and not reviewable.

## Consequences

- Every training run has a portable event stream next to its artifacts even
  when MLflow is not installed.
- Report collection is a separate, deterministic step; T11 validates it with
  `scripts/validate_reports.py experiments`.
- Tracker failures are visible through `CompositeTracker.errors` instead of
  crashing a multi-hour run.

## Evidence

- `uv run pytest tests/unit/adapters/test_trackers.py -q`
- `uv run pytest tests/unit/application/test_tracker_factory.py -q`
- `uv run pytest tests/unit/scripts/test_collect_experiment_reports.py -q`
