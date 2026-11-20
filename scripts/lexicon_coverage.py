#!/usr/bin/env python
"""Print how many utterances each regionalisation profile changes (S9).

Only dev or train may be inspected: the test split stays behind the audited
evaluation path, so ``--split test`` exits 6 without reading anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.domain.examples import Example
from stc.domain.perturbations import regionalize
from stc.domain.splits import Split

PROFILES = ("mx", "ec", "ar")


def _default_data_dir() -> Path:
    configured = os.environ.get("STC_DATA_DIR")
    if configured:
        return Path(configured)
    repository_data = Path("data") / "processed" / "massive-1.0"
    if repository_data.is_dir():
        return Path("data")
    from stc.config import Settings

    return Settings().resolve_data_dir()


def _load(split: Split, data_dir: Path) -> Sequence[Example]:
    repository = JsonlDatasetRepository(data_dir / "processed" / "massive-1.0")
    return DevOnlyRepository(repository).load(split)


def _changed(examples: Sequence[Example], profile: str) -> int:
    return sum(1 for example in examples if regionalize(example.text, profile) != example.text)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the coverage report; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=[split.value for split in Split],
        default=Split.DEV.value,
        help="split to inspect (dev or train; test is refused)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="processed data root (defaults to STC_DATA_DIR or ./data)",
    )
    args = parser.parse_args(argv)
    split = Split(args.split)
    if split is Split.TEST:
        print(
            "error: the test split is locked; lexicon coverage runs on dev or train only",
            file=sys.stderr,
        )
        return 6
    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir()
    examples = _load(split, data_dir)
    print(f"split {split.value}: {len(examples)} utterances")
    for profile in PROFILES:
        print(f"{profile} {_changed(examples, profile)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
