# Contributing

Thanks for taking a look. This repository is a local-first portfolio project, so the
contribution workflow is deliberately small and reproducible.

## Development setup

Requirements: Python 3.12 (managed by `uv`), git, and optionally an NVIDIA GPU for the
training tests.

```bash
git clone https://github.com/Eduardo12Pacheco/spanish-text-classifier-lora.git
cd spanish-text-classifier-lora
uv sync --locked --group dev
uv run stc --version
```

`uv.lock` is committed. Update it deliberately with `uv lock` and explain dependency
changes in the pull request.

## The gate

Every change must pass the same gate that CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest -q -m "not e2e and not torch"
```

Add coverage when the change touches `src/stc`:

```bash
uv run pytest -q --cov=src/stc --cov-fail-under=85
```

Training tests and browser tests are opt-in because they need extra packages:

```bash
uv sync --locked --extra train --group dev
uv run pytest -q -m torch

uv run playwright install chromium
uv run pytest -q -m e2e
```

## Test-first and boundaries

- Write the failing test first, then the smallest implementation that passes.
- Keep the hexagon intact: `domain` is pure Python plus NumPy, `application` owns ports
  and use cases, adapters hold I/O. `lint-imports` and
  `tests/architecture/test_layers.py` enforce the direction.
- Do not weaken, skip or delete a test to get a green run.
- Run `uv run python scripts/check_no_weights.py` before committing: weights, datasets
  and large archives never enter git.

## Commits

Conventional Commits in English, one coherent unit per commit (roughly 400 changed lines
or fewer). Include tests and docs with the behaviour they describe. Do not add AI
attribution lines. Branch from `main`; do not force-push.

## Releases

Releases are tag-driven and owned by the maintainer: push `main`, tag `vX.Y.Z`, let CI
build the wheel and sdist, then attach the model bundle with `scripts/publish_model.sh`
when the release task calls for it.
