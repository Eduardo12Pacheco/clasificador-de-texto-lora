# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub private
vulnerability reporting ("Report a vulnerability" in the Security tab) on
`Eduardo12Pacheco/spanish-text-classifier-lora`, or the repository's private Security
Advisory form. Include affected version, reproduction steps and impact. We aim to
acknowledge reports within seven days.

## Threat model (initial)

- **Untrusted model bundles.** Release downloads are verified against the committed
  `src/stc/model_registry.json` checksum; extracted bundles are re-verified per file
  through `manifest.json`; archive extraction rejects path traversal. `skops` loading
  uses a type allow-list rather than pickle defaults.
- **Local server.** `stc serve` binds `127.0.0.1` by default, serves static assets from
  the package and sends security headers. It is a local tool, not a hardened public
  service.
- **No telemetry.** The project performs no analytics or background network calls.
  Dataset and model downloads are explicit commands with pinned URLs and checksums.
- **Input logging.** Request text is never logged; only lengths and metadata are.
- **Supply chain.** Dependencies are locked with `uv.lock`, audited with `pip-audit`,
  and the history is scanned with gitleaks.

This section is finalised together with the release pipeline; report anything that looks
under-specified rather than assuming it is safe.
