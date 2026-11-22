# Contributing

Thanks for taking a look. This repository is a local-first portfolio project, so the
contribution workflow is deliberately small and reproducible.

## Development setup

Requirements: Python 3.12 (managed by `uv`), git, and optionally an NVIDIA GPU for the
training tests.

```bash
git clone https://github.com/Eduardo12Pacheco/clasificador-de-texto-lora.git
cd clasificador-de-texto-lora
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

Read `docs/ARCHITECTURE.md` before changing a layer and `docs/EVALUATION.md`
before changing a metric or a split rule. The GPU pipeline is documented in
`docs/TRAINING.md`.

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

Releases are tag-driven and the model assets are attached manually. Never commit or
upload weights from CI.

1. Confirm the version in `pyproject.toml` and the model bundle in
   `src/stc/model_registry.json` agree with the release you are cutting.
2. Push `main`, then create and push the annotated tag:

   ```bash
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin main
   git push origin v0.1.0
   ```

3. The `Release` workflow checks that the tag matches the package version, builds the
   wheel and sdist, generates release notes from Conventional Commits with git-cliff,
   writes `SHA256SUMS` and creates the GitHub Release. It refuses any asset above
   100 MB.
4. Attach the verified model bundle to that release:

   ```bash
   scripts/publish_model.sh v0.1.0
   ```

   The helper requires an authenticated `gh`, an existing release, and a bundle whose
   sha256 equals the registry entry. It uploads `dist-model/stc-es-intent-*.zip` and
   `dist-model/SHA256SUMS-model.txt`; it never touches git.
5. The `Verify release` workflow runs on the published release (and can be dispatched
   with a tag): it installs the released wheel on Linux, Windows and macOS, waits for
   the model asset, pulls and verifies the bundle, checks
   `tests/fixtures/smoke_phrases.json` and runs `stc doctor`.

Local installation behaves like the released package:

```bash
bash scripts/smoke_install.sh --check   # prerequisites only
bash scripts/smoke_install.sh           # build, install, exercise, verify torch is absent
bash scripts/smoke_install.sh --uninstall
pwsh ./scripts/smoke_install.ps1 -Check # Windows equivalent
```

The smoke installation uses a temporary `UV_TOOL_DIR` and `STC_HOME`, so it never
replaces your normal `stc`. CI runs the same scripts in the `smoke-install` matrix job.

Security-relevant checks before a release: `gitleaks git --no-banner`, the dependency
audit described in `SECURITY.md`, and `uv run python scripts/validate_workflows.py`.
