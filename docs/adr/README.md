# Architecture decision records

This directory records the decisions that shape the project. Each ADR is
immutable once accepted; later changes are appended as an addendum or a new ADR.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-hexagonal-architecture.md) | Hexagonal architecture with enforced dependency direction | Accepted |
| [0002](0002-task-and-dataset-selection.md) | Task and dataset selection | Accepted |
| [0004](0004-evaluation-protocol-and-leakage-policy.md) | Evaluation protocol and leakage policy | Accepted |
| [0006](0006-calibration-and-abstention.md) | Calibration and abstention | Accepted |

ADR-0003 (backbone and adaptation method) is introduced by the training task;
0005 and 0007–0009 follow with inference, distribution, tracking and the
synthetic robustness suite. Every ADR uses the template below.

## Template

```markdown
# ADR-NNNN — Title

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD
- **Deciders:** project owner

## Context

What forces are at play? What is the problem?

## Decision

What we chose, stated so that it can be verified.

## Alternatives

What else was considered and why it was rejected.

## Consequences

What becomes easier, what becomes harder, and what must be revisited.

## Evidence

Commands, tests, measurements or documents that prove the decision is in force.
```
