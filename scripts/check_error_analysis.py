#!/usr/bin/env python
"""Check ``docs/ERROR_ANALYSIS.md`` against the audited error list (S10).

Every ``ex-<id>`` cited in the document must exist in ``reports/final/errors.json``.
The document needs at least five ``###`` categories and at least three real
example ids in each one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

EXAMPLE_ID = re.compile(r"\bex-([A-Za-z0-9_-]+)\b")
MIN_CATEGORIES = 5
MIN_IDS_PER_CATEGORY = 3


def _load_errors(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    rows = payload.get("errors")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: errors must be a list")
    known: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("example_id"), str):
            raise ValueError(f"{path}: every error needs a string example_id")
        known.add(str(row["example_id"]))
    return known


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line[4:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


def check(document: str, known_ids: set[str]) -> list[str]:
    """Return every violation found in ``document``."""
    problems: list[str] = []
    sections = _sections(document)
    if len(sections) < MIN_CATEGORIES:
        problems.append(
            f"expected at least {MIN_CATEGORIES} '### ' categories, found {len(sections)}"
        )
    for title, body in sections:
        cited = EXAMPLE_ID.findall(body)
        if len(set(cited)) < MIN_IDS_PER_CATEGORY:
            problems.append(
                f"category {title!r} cites {len(set(cited))} example ids, "
                f"expected at least {MIN_IDS_PER_CATEGORY}"
            )
        for example_id in sorted(set(cited)):
            if example_id not in known_ids:
                problems.append(f"category {title!r} cites unknown example id ex-{example_id}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Run the error-analysis check; return 0 on success and 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=Path("docs/ERROR_ANALYSIS.md"))
    parser.add_argument("--errors", type=Path, default=Path("reports/final/errors.json"))
    args = parser.parse_args(argv)
    try:
        known = _load_errors(args.errors)
        problems = check(args.doc.read_text(encoding="utf-8"), known)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    print(f"OK error analysis: {args.doc} cites only real example ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
