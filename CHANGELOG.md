# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-20

### Added

- Leakage-audited MASSIVE 1.0 es-ES pipeline (10,898 / 2,033 / 2,974 rows) with a
  committed fixture sample, checksums and a reproducible archived download.
- Strong TF-IDF + LinearSVC baseline selected on dev from the D5 grid, plus a
  LoRA fine-tune of `BSC-LT/MrBERT-es` selected by a 24-trial Optuna search and
  three final seeds; screening and full fine-tuning references are committed.
- Audited final comparison on the locked test split: accuracy, macro-F1 (95 %
  bootstrap CIs), exact McNemar, per-class and confusion reports, calibration,
  selective accuracy, robustness and slice analysis, and a full error list.
- Temperature scaling and a dev-chosen review threshold exposed as
  `needs_review` by the CLI, the FastAPI service and the web UI.
- ONNX export with fp32 parity and the D12 quantisation gate; every int8
  variant failed the gate, so the CPU bundle ships fp32 with `onnxruntime`.
- Typer CLI (`stc data|train|export|calibrate|bundle|model|evaluate|report|
  predict|benchmark|serve|doctor`), FastAPI service and a local ES/EN web UI
  with light/dark themes, empty/loading/error/offline states and Playwright
  coverage.
- Nine ADRs, a generated model card, an error-analysis document, GitHub Release
  model distribution with sha256 verification, and a smoke-phrase fixture for
  release verification.
- Initial uv project scaffold with a typed package layout, hexagon boundaries
  enforced by import-linter and an AST test, and a Typer CLI entrypoint.
- Quality/test GitHub Actions skeleton, repository artifact policy and workflow
  validation.
- Architecture, evaluation, training and data guides, a generated CLI/API
  reference, and tests that guard links, ADR structure, quoted paths and the
  documented training commands.

### Changed

- Static QDQ uses MinMax calibration instead of percentile because percentile
  per-channel histograms need ~19 GiB RSS; documented in ADR-0005.

### Fixed

- Test-access audit entries now include the sha256 of the scored model.
- The tokenizer adapter falls back to the decoder end-of-text token when an
  encoder `<pad>` token is absent.
- Screening releases the previous backbone before loading the next candidate and
  the OOM retry no longer keeps the failed model's GPU tensors alive.

