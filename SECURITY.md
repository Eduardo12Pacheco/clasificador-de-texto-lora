# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub private
vulnerability reporting ("Report a vulnerability" in the Security tab) on
`Eduardo12Pacheco/clasificador-de-texto-lora`, or the repository's private Security
Advisory form. Include the affected version, reproduction steps and impact. We aim to
acknowledge reports within seven days.

## Threat model

The project is a local-first tool. It downloads a public dataset and a public model
bundle on explicit commands, serves a UI on loopback and never phones home. The
following assets and boundaries are in scope.

### Untrusted model bundles

- `src/stc/model_registry.json` pins the release URL, size and sha256 of the published
  bundle. `stc model pull` streams the download to a temporary file and refuses to
  install it when the checksum or size differs (`IntegrityError`, exit 3).
- Every unpacked bundle is checked against its `manifest.json`: each listed file must
  match its recorded sha256 and size, unexpected files are rejected, and unsafe paths
  (absolute paths, `..`, backslashes, symlinks) are refused before extraction.
- `stc model verify` re-checks an installed bundle at any time.
- Baseline `skops` artifacts are loaded through a type allow-list restricted to
  scikit-learn, NumPy, SciPy and builtin containers; any other type raises
  `UntrustedArtifactError` (exit 3) instead of unpickling it.
- The ONNX Runtime session never enables custom operators or remote model fetching.

### Local server

- `stc serve` binds `127.0.0.1:19080` by default (configurable with `STC_API_HOST` and
  `STC_API_PORT`; the port range 19000-19999 avoids collisions with the other local
  projects on this machine). It prints a warning when asked to bind a non-loopback
  interface and is not a hardened public service.
- Responses carry `Content-Security-Policy: default-src 'self'`,
  `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. The UI makes no
  third-party requests and loads no remote fonts or scripts.
- Input text is never logged; request logs contain only lengths, identifiers and
  timings. Validation bounds (64 texts, 512 characters each, `top_k` 1-10) limit
  accidental resource exhaustion.

### Supply chain and repository hygiene

- Dependencies are locked in `uv.lock` and audited in CI with
  `uv export --locked --no-dev ... | uvx pip-audit --no-deps`.
- The full git history is scanned with gitleaks. GitHub Actions references are checked
  by `scripts/validate_workflows.py` and actionlint, and workflows request the minimum
  `permissions`.
- Model weights, datasets and archives never enter git: `scripts/check_no_weights.py`
  fails on tracked files above 1 MB or with a weight/bundle extension outside the tiny
  test fixtures.
- Releases are tag-driven. CI builds only the wheel and sdist and refuses assets above
  100 MB; the verified model bundle is attached afterwards by the maintainer with
  `scripts/publish_model.sh`, which checks the registry checksum first.

## Dependency audit exceptions

There are currently no ignored advisories. If an advisory cannot be fixed immediately,
the exact package, advisory identifier, justification and review date are recorded here
and in the commit that adds the ignore, rather than silently disabling the audit.

## Running the checks locally

```bash
uv sync --locked --group dev
uv run python scripts/check_no_weights.py
uv run python scripts/validate_workflows.py
uv export --locked --no-dev --no-hashes --format requirements-txt -o /tmp/req.txt
uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip
gitleaks git --no-banner
```

Release and installer steps are documented in `CONTRIBUTING.md` and exercised by the
`smoke-install` CI job.
