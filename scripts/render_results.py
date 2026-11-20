#!/usr/bin/env python
"""Build ``reports/summary.json`` from the final reports and render results.

The rendering logic lives in :mod:`stc.adapters.reporting.results` so the CLI
(``stc report render``) and this script share one implementation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stc.adapters.reporting.results import render_results


def main(argv: Sequence[str] | None = None) -> int:
    """Render the summary, table and optional README blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/summary.json"))
    parser.add_argument("--table", type=Path, default=Path("reports/results_table.md"))
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--readme-es", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        render_results(
            summary_path=args.summary,
            table_path=args.table,
            final_dir=args.final_dir,
            readme=args.readme,
            readme_es=args.readme_es,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.summary} and {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
