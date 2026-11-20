#!/usr/bin/env python3
"""Check that relative Markdown links and heading anchors resolve (S19)."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from urllib.parse import unquote

SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        ".cache",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "artifacts",
        "build",
        "dist",
        "dist-model",
        "htmlcov",
        "node_modules",
        "__pycache__",
    }
)
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "tel:")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
INLINE_LINK_RE = re.compile(
    r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(?:<([^>]+)>|([^)\s]+))(?:\s+[^)]*)?\)"
)
HTML_ANCHOR_RE = re.compile(r"""<a\s+[^>]*(?:name|id)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
CODE_SPAN_RE = re.compile(r"`([^`]*)`")


def markdown_files(root: Path) -> list[Path]:
    """Return every Markdown file below ``root``, excluding generated trees."""
    found: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIPPED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            found.append(path)
    return found


def _strip_fences(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` for lines outside fenced code blocks."""
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if fence is None and match:
            fence = match.group(1)
            continue
        if fence is not None:
            if match and match.group(1) == fence:
                fence = None
            continue
        yield number, line


def slugify_heading(text: str) -> str:
    """Approximate GitHub's heading-anchor slug for Markdown text."""
    cleaned = CODE_SPAN_RE.sub(r"\1", text)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = cleaned.replace("*", "").replace("_", "").strip()
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    return re.sub(r"\s+", "-", cleaned).strip("-")


def anchors(path: Path) -> set[str]:
    """Return the heading and explicit HTML anchors defined by one document."""
    text = path.read_text(encoding="utf-8")
    found: set[str] = set()
    seen: dict[str, int] = {}
    previous: str | None = None
    for _number, line in _strip_fences(text):
        heading_match = ATX_HEADING_RE.match(line)
        if heading_match:
            title = heading_match.group(2)
        elif previous is not None and SETEXT_UNDERLINE_RE.match(line):
            title = previous
        else:
            title = None
            previous = line if line.strip() else None
            continue
        slug = slugify_heading(title)
        if slug in seen:
            seen[slug] += 1
            found.add(f"{slug}-{seen[slug]}")
        else:
            seen[slug] = 0
            found.add(slug)
        previous = None
        if heading_match:
            continue
    for match in HTML_ANCHOR_RE.finditer(text):
        found.add(match.group(1))
    return found


def links_in(path: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, target)`` for every local inline link in ``path``."""
    results: list[tuple[int, str]] = []
    for number, line in _strip_fences(path.read_text(encoding="utf-8")):
        for match in INLINE_LINK_RE.finditer(line):
            target = match.group(1) or match.group(2)
            if target:
                results.append((number, target))
    return results


def _is_external(target: str) -> bool:
    lowered = target.lower()
    return lowered.startswith(EXTERNAL_SCHEMES)


def _resolve(root: Path, path: Path, target: str) -> tuple[Path, str] | str:
    """Resolve one link, returning ``(file, anchor)`` or an error string."""
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    decoded = unquote(target)
    if not decoded:
        return "empty link target"
    if decoded.startswith("#"):
        return path, decoded[1:]
    if decoded.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", decoded):
        return f"absolute path is not portable: {decoded}"
    file_part, _, fragment = decoded.partition("#")
    if not file_part:
        return path, fragment
    candidate = (path.parent / file_part).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return f"link escapes the repository: {decoded}"
    if not candidate.exists():
        return f"missing target: {decoded}"
    return candidate, fragment


def check_file(root: Path, path: Path, cache: dict[Path, set[str]]) -> list[str]:
    """Return violations for every local link in one Markdown file."""
    errors: list[str] = []
    for number, target in links_in(path):
        if _is_external(target) or target.startswith("#!"):
            continue
        resolved = _resolve(root, path, target)
        if isinstance(resolved, str):
            errors.append(f"{path.relative_to(root)}:{number}: {resolved}")
            continue
        candidate, fragment = resolved
        if candidate.is_dir():
            continue
        if fragment and candidate.suffix.lower() == ".md":
            known = cache.setdefault(candidate, anchors(candidate))
            if fragment not in known:
                errors.append(
                    f"{path.relative_to(root)}:{number}: missing anchor #{fragment} "
                    f"in {candidate.relative_to(root)}"
                )
    return errors


def check_tree(root: Path, files: Sequence[Path] | None = None) -> list[str]:
    """Return every broken local link under ``root``."""
    selected = list(files) if files is not None else markdown_files(root)
    cache: dict[Path, set[str]] = {}
    errors: list[str] = []
    for path in selected:
        errors.extend(check_file(root, path, cache))
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    """Validate Markdown links and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=None,
        help="Check only this file (repeatable).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    files = None
    if args.file:
        files = [(path if path.is_absolute() else root / path).resolve() for path in args.file]
    errors = check_tree(root, files)
    if errors:
        print("Broken Markdown links:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    checked = len(files) if files is not None else len(markdown_files(root))
    print(f"OK: {checked} Markdown file(s) checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
