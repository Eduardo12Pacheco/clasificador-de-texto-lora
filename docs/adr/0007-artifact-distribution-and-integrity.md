# ADR-0007 — Artifact distribution, bundles and integrity

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

The model is ~574 MiB in fp32, far beyond what belongs in git, and the project
is distributed as a Python tool with no hosted service. Users must be able to
obtain the exact bytes the reports describe, detect a corrupt or tampered
download, and do so offline. The baseline artifact is a pickled sklearn object,
which is a code-execution risk if loaded from an untrusted source.

## Decision

1. **Bundle layout (`bundle_version: 1`, D13).** A directory containing
   `manifest.json`, `model.onnx` (or the shipped variant), `tokenizer.json`,
   `labels.json`, `calibration.json`, optional `baseline/`, `MODEL_CARD.md`,
   `LICENSE` and `NOTICE`. The manifest records bundle and tool versions, model
   parameters, base-model id/revision/licence, adapter recipe, dataset id and
   counts, ONNX op-set and input/output names, tokenizer settings, and a
   `files` list with `{path, sha256, size}` for every content file.
2. **Manifest as trust anchor.** `verify_bundle` hashes every listed file,
   rejects size or checksum mismatches, rejects unsafe relative paths, and
   reports unexpected files instead of silently ignoring them. Loading a
   bundle verifies it by default; the CLI's explicit variant path skips the
   bundle scan because quantisation variants live outside the bundle.
3. **Deterministic releases.** `stc bundle pack` re-hashes the assembled
   layout, writes sorted zip entries with a fixed 1980-01-01 timestamp and
   deflate level 9, and prints the sha256 and size. Two packs of the same inputs
   are byte-identical. The LoRA adapter ships as a second archive.
4. **Committed registry, external weights.** `src/stc/model_registry.json`
   stores only `{name, version, url, sha256, size_bytes, license}`. `stc model
   pull` downloads with `urllib`, streams to a temporary file, verifies the
   sha256, and extracts with a zip-slip guard (rejecting absolute paths, `..`,
   backslashes and symlinks). `STC_OFFLINE=1` refuses network pulls with exit 5;
   `--from-file` still works offline. Installed bundles live under
   `${STC_HOME}/models/<name>/<version>/`.
5. **skops allow-list.** The baseline is loaded through `skops.io` with a type
   allow-list restricted to sklearn, numpy, scipy.sparse and builtins; any other
   type raises `UntrustedArtifactError` (exit 3). Its own manifest hashes
   `baseline.skops` and `labels.json`.
6. **Distribution transport.** The bundle is a GitHub Release asset attached by
   `scripts/publish_model.sh` (T13), never by CI and never in git; GitHub
   allows assets below 2 GiB. `scripts/check_no_weights.py` fails if weights or
   archives become tracked.

## Alternatives

- Git LFS: still makes clones heavy and couples weights to history.
- Hugging Face Hub in v0.1.0: optional later, not needed for a local-first tool.
- A raw tarball without per-file hashes: no way to detect a partially replaced
  bundle.
- `pickle` for the baseline: arbitrary code execution; rejected in favour of
  skops plus the allow-list.

## Consequences

- Every installed model is reproducible from the registry entry and verifiable
  offline.
- Packing is deterministic, so the registry sha256 can be compared by a human
  or a CI step.
- Changing the bundle layout requires a `bundle_version` bump and a new ADR
  addendum.
- A corrupt download fails before anything is installed; partial state cannot
  shadow a good bundle because installation copies into a fresh tree.

## Evidence

- `uv run pytest tests/unit/adapters/test_bundle_io.py -q`
- `uv run pytest tests/unit/adapters/test_model_hub.py -q`
- `uv run pytest tests/unit/adapters/test_registry_and_store.py -q`
- `uv run stc model verify tests/fixtures/tiny_bundle`
- `uv run stc bundle pack --bundle tests/fixtures/tiny_bundle --out "$(mktemp -d)/a.zip"`
