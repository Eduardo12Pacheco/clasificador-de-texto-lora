#!/usr/bin/env python
"""Build the deterministic CC BY 4.0 sample bundled with the package (D24)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.config import Settings
from stc.domain.examples import Example
from stc.domain.splits import Split

MAX_PER_INTENT = {Split.TRAIN: 6, Split.DEV: 2, Split.TEST: 2}
MAX_BYTES = 120 * 1024

_NOTICE_TEMPLATE = """\
Amazon MASSIVE
==============

{amazon_notice}

Sample selection and reformatting
---------------------------------
The JSONL files in this directory are a small sample selected and reformatted
by spanish-text-classifier-lora, 2026. Rows are drawn from the cleaned training
split and the official development/test splits of Amazon MASSIVE 1.0 es-ES,
licensed CC BY 4.0. The sample keeps up to {train} train, {dev} dev and {test}
test rows per intent, chosen deterministically by ascending sha256(id).
"""


def _select(rows: list[Example], limit: int) -> list[Example]:
    ordered = sorted(rows, key=lambda example: hashlib.sha256(example.id.encode()).hexdigest())
    return ordered[:limit]


def _write_split(path: Path, rows: list[Example]) -> None:
    lines = [
        json.dumps(
            {
                "id": example.id,
                "text": example.text,
                "intent": example.intent,
                "scenario": example.scenario,
                "worker_id": example.worker_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for example in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_fixture(data_dir: Path, out_dir: Path) -> dict[str, int]:
    """Select the sample from processed splits and write the fixture directory."""
    repository = JsonlDatasetRepository(data_dir / "processed" / "massive-1.0")
    selected: dict[Split, list[Example]] = {}
    for split in Split:
        by_intent: dict[str, list[Example]] = defaultdict(list)
        for example in repository.load(split):
            by_intent[example.intent].append(example)
        chosen = [_select(rows, MAX_PER_INTENT[split]) for rows in by_intent.values()]
        selected[split] = sorted(
            (example for group in chosen for example in group),
            key=lambda example: (example.intent, example.id),
        )

    intents = {example.intent for example in selected[Split.TRAIN]}
    if len(intents) != 60:
        raise SystemExit(f"fixture train must cover all 60 intents, found {len(intents)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, rows in selected.items():
        _write_split(out_dir / f"{split.value}.jsonl", rows)
        counts[split.value] = len(rows)

    raw_dir = data_dir / "raw" / "massive-1.0"
    (out_dir / "LICENSE-CC-BY-4.0.txt").write_bytes((raw_dir / "LICENSE").read_bytes())
    amazon_notice = (raw_dir / "NOTICE.md").read_text(encoding="utf-8").strip()
    (out_dir / "NOTICE.md").write_text(
        _NOTICE_TEMPLATE.format(
            amazon_notice=amazon_notice,
            train=MAX_PER_INTENT[Split.TRAIN],
            dev=MAX_PER_INTENT[Split.DEV],
            test=MAX_PER_INTENT[Split.TEST],
        ),
        encoding="utf-8",
        newline="\n",
    )

    total_bytes = sum(path.stat().st_size for path in out_dir.iterdir() if path.is_file())
    if total_bytes > MAX_BYTES:
        raise SystemExit(f"fixture is {total_bytes} bytes, over the {MAX_BYTES}-byte budget")
    return counts


def main() -> None:
    """Parse CLI flags and build the sample."""
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.resolve_data_dir())
    default_out = (
        Path(__file__).resolve().parents[1] / "src" / "stc" / "_fixture" / "massive_es_sample"
    )
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    counts = build_fixture(args.data_dir, args.out)
    print(
        f"fixture written to {args.out}: train={counts['train']} dev={counts['dev']} "
        f"test={counts['test']}"
    )


if __name__ == "__main__":
    main()
