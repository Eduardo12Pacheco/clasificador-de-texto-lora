# ADR-0002 — Task and dataset selection

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

The project needs a Spanish utterance-classification task that is small enough
to train on a single 8 GB GPU, large enough for meaningful per-class and
calibration analysis, and licensed so that code, metrics and a redistributable
sample can be published. Most Spanish datasets found during research were
either non-commercial, licence-ambiguous, built from automatically derived
labels, or too small for a 60-way study.

## Decision

Use **Amazon MASSIVE 1.0, locale `es-ES`** (CC BY 4.0), a human-localised
Spanish version of SLURP:

- primary label: `intent` (60 classes, `scenario_intent` format);
- secondary label: `scenario` (18 classes, the intent prefix before the first
  underscore, verified equal to MASSIVE's `scenario` field for all 16,521 rows);
- label order: intents sorted lexicographically and stored in `labels.json`;
- partitions: the official train/dev/test splits (11,514 / 2,033 / 2,974 rows),
  with the train-only cleaning policy of ADR-0004.

The pinned tarball, its size and sha256 live in `src/stc/adapters/data/massive_archive.py`
and are documented in `docs/DATA.md`. Only three archive members are extracted
(`es-ES.jsonl`, `LICENSE`, `NOTICE.md`); every other member is ignored and any
unsafe member rejects the archive.

## Alternatives

| Candidate | Why rejected |
|---|---|
| MeLiSA (Mercado Libre reviews, 8 LatAm countries) | Licence "other" tied to an Amazon agreement, demo-only release; not redistributable. |
| TASS / InterTASS | No licence statement found; research-only tweet distribution. |
| MLSUM (es) | Non-commercial terms; El País (Spain) only. |
| XL-Sum | CC BY-NC-SA 4.0; non-commercial. |
| SIB-200 | Seven topic classes; too small for a per-class study. |
| WikiCAT_es v2 | Labels derived automatically from Wikipedia categories; inconsistent licence metadata. |
| MTEB SpanishNewsClassification | Upstream provenance cannot be verified. |
| cardiffnlp tweet sentiment | No licence metadata; tweets. |

There is no open-licensed, redistributable Spanish corpus with natural
Latin-American dialect labels, so the project does not claim LatAm accuracy;
it probes dialect sensitivity with a synthetic perturbation suite (ADR-0009).

## Consequences

- The task is a realistic 60-way imbalanced classification problem: 4 to 810
  train rows per class (`cooking_query` has four train rows and none in dev/test).
- The dataset contains real cross-split duplicates; the audit and cleaning
  policy are mandatory and are recorded in `reports/data_audit.json`.
- Spanish is the `es-ES` locale only. Results must not be presented as evidence
  for Latin-American Spanish, and the synthetic dialect probe is labelled as
  such in the README, model card and reports.
- Redistributing a small sample is allowed with attribution; the bundled
  fixture carries `LICENSE-CC-BY-4.0.txt` and `NOTICE.md`.

## Evidence

- `uv run stc data fetch` → `sha256 OK 7df623fd…153f8`, 16,521 rows, 60 intents,
  18 scenarios.
- `uv run stc data audit --out reports` → `reports/data_audit.json` reproduces
  dev∩test 45, dev-in-train 123, test-in-train 152.
- `uv run stc data fixture info` → 358 train / 118 dev / 117 test rows.
