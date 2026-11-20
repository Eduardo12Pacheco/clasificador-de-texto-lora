# ADR-0001 — Hexagonal architecture with enforced dependency direction

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner

## Context

`spanish-text-classifier-lora` grows from a pure classification model into a data
pipeline, a training stack, a reporting suite, a CLI, an HTTP API and a local web
UI. Those concerns change at very different rates: label rules and leakage policy
should stay stable and trivially testable, while framework and library choices
(Typer, FastAPI, torch, ONNX Runtime) are replaceable details. Mixing them makes
the test split lock, the leakage policy and the metrics hard to audit.

## Decision

Use a hexagonal (ports & adapters) architecture with three layers:

- `stc.domain` — pure value objects and rules (labels, text normalisation,
  leakage, metrics, calibration). It imports only the standard library, `numpy`
  and other `stc.domain` modules. No file system, network, framework or clock.
- `stc.application` — use cases, ports (`typing.Protocol`) and DTOs expressed as
  dataclasses. It orchestrates the domain and depends only on abstractions.
- `stc.adapters` — concrete driven and driving adapters (data, models,
  inference, training, export, tracking, reporting, hub, storage, CLI, API, web).

`stc.config`, `stc.logging` and `stc.bootstrap` sit outside the layers.
`bootstrap` is the only composition root: adapters never import each other and
all wiring happens there, so the dependency direction is acyclic.

The test split is protected by a capability token (`TestAccessGrant`) issued
only to the audited final-evaluation use cases. This keeps the "read the test
split once" rule a property of the domain, not a convention in scripts.

## Alternatives

- **Classic three-layer MVC** — mapping controllers/models/services invites
  framework imports in the core and gives no enforceable boundary; the leakage
  and test-lock rules would be untestable in isolation.
- **Single-package script-style layout** — fastest to start, but makes the
  training/export/inference adapters impossible to swap and the domain
  untestable without heavy dependencies installed.
- **No enforced structure, documented conventions only** — conventions rot;
  the repository already needs `numpy`-only domain tests and a torch-free
  runtime path, which are easy to violate accidentally.

## Consequences

- Domain tests run in milliseconds with no optional extras installed.
- The CPU inference path cannot accidentally import `torch`: a contract fails
  instead of a user discovering it at runtime.
- Some indirection is unavoidable: use cases talk to protocols and DTOs, and
  adapters are wired in `bootstrap`. The indirection is limited to real seams
  (dataset source, classifier, exporter, tracker, model store, clock).
- A new adapter must be registered in `bootstrap`, never imported by another
  adapter.

## Evidence

Enforced mechanically, not by review:

- `[tool.importlinter]` in `pyproject.toml` declares six contracts: layer
  direction, domain isolation (no outer layer, no third-party libraries),
  application isolation, a torch-free runtime package set, and adapter
  independence. `uv run lint-imports` must print `0 broken`.
- `tests/architecture/test_layers.py` runs `lint-imports` and additionally walks
  the AST of every `src/stc/domain/*.py` module, asserting that it imports only
  the standard library, `numpy` and `stc.domain`.
- `tests/unit/domain/` exercises the pure rules without any adapter, database,
  network or optional dependency.
