#!/usr/bin/env python
"""Export the FastAPI OpenAPI document as a stable committed snapshot (S14)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from stc.adapters.api.app import create_app
from stc.adapters.api.provider import LoadedModel
from stc.config import Settings


class _UnusedProvider:
    """Provider that is never called during OpenAPI generation."""

    def load_default(self) -> LoadedModel:
        raise RuntimeError("the OpenAPI snapshot does not load a model")

    def load_baseline(self) -> LoadedModel:
        raise RuntimeError("the OpenAPI snapshot does not load a model")

    def baseline_available(self) -> bool:
        return False


def build_openapi() -> Mapping[str, object]:
    """Return the deterministic OpenAPI document of the real app factory."""
    application = create_app(Settings(), _UnusedProvider())
    document = application.openapi()
    if not isinstance(document, dict):
        raise TypeError("FastAPI returned a non-mapping OpenAPI document")
    return document


def dump_openapi(document: Mapping[str, object]) -> str:
    """Serialise the OpenAPI document with sorted keys and a trailing newline."""
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write ``docs/openapi.json`` or the path given by ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/openapi.json"))
    args = parser.parse_args(argv)
    try:
        text = dump_openapi(build_openapi())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    except (OSError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
