# Architecture decision records

This directory records the decisions that shape the project. Each ADR is
immutable once accepted; later changes are appended as an addendum or a new ADR.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-hexagonal-architecture.md) | Hexagonal architecture with enforced dependency direction | Accepted |

ADRs 0002–0009 are introduced by the tasks that make the corresponding
decisions (dataset selection, backbone, evaluation protocol, inference,
calibration, distribution, experiment tracking, synthetic robustness).

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
