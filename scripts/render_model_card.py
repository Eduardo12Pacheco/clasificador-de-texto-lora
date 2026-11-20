#!/usr/bin/env python
"""Render ``MODEL_CARD.md`` from the summary and a template (T12 step 8).

Every placeholder is ``{{dotted.path}}`` and is resolved only from
``reports/summary.json``; an unknown path is an error, never an empty string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.:-]+)\}\}")


def _load(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _resolve(payload: Mapping[str, object], path: str) -> object:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"unknown placeholder: {{{{{path}}}}}")
        current = current[part]
    return current


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(template: str, summary: Mapping[str, object]) -> str:
    """Fill every placeholder or raise listing the unknown paths."""
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        try:
            return _format(_resolve(summary, path))
        except ValueError:
            unknown.append(path)
            return match.group(0)

    rendered = PLACEHOLDER.sub(replace, template)
    if unknown:
        raise ValueError(f"unknown placeholder(s): {', '.join(sorted(set(unknown)))}")
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    """Render the model card; return 0 on success and 1 on a template error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("docs/model_card.template.md"))
    parser.add_argument("--summary", type=Path, default=Path("reports/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("MODEL_CARD.md"))
    args = parser.parse_args(argv)
    try:
        rendered = render(
            args.template.read_text(encoding="utf-8"),
            _load(args.summary),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
