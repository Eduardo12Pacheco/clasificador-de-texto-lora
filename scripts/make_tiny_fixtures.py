#!/usr/bin/env python
"""Build the deterministic tiny artifacts used by the torch-marked tests.

The T7 artifact is ``tests/fixtures/tiny/tokenizer.json``: a WordLevel
tokenizer whose vocabulary comes from the bundled MASSIVE sample. T8 extends
this script with the tiny ONNX bundle. Everything is generated from committed
inputs, so running the script twice must produce byte-identical files.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "src" / "stc" / "_fixture" / "massive_es_sample"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "tiny"

PAD_TEXT = "<pad>"
UNK_TEXT = "<unk>"


def _fixture_texts() -> list[str]:
    texts: list[str] = []
    for split in ("train", "dev", "test"):
        path = FIXTURE_DIR / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                texts.append(str(payload["text"]))
    return texts


def vocabulary_from_texts(texts: Iterable[str]) -> dict[str, int]:
    """Build the sorted WordLevel vocabulary with the two special tokens first."""
    words = sorted({word for text in texts for word in text.split()})
    vocab = {PAD_TEXT: 0, UNK_TEXT: 1}
    for index, word in enumerate(words, start=2):
        vocab[word] = index
    return vocab


def build_tokenizer_payload(texts: Iterable[str]) -> dict[str, object]:
    """Return the tokenizers JSON document for the tiny WordLevel tokenizer."""
    vocab = vocabulary_from_texts(texts)
    added = [
        {"id": 0, "content": PAD_TEXT, "special": True},
        {"id": 1, "content": UNK_TEXT, "special": True},
    ]
    for token in added:
        token.update(
            {
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
        )
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": UNK_TEXT,
        },
    }


def write_json(path: Path, payload: object) -> None:
    """Write deterministic, newline-terminated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def build_tiny_fixtures(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    """Generate every tiny fixture and return the paths written."""
    payload = build_tokenizer_payload(_fixture_texts())
    tokenizer_path = Path(out_dir) / "tokenizer.json"
    write_json(tokenizer_path, payload)
    return [tokenizer_path]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the tiny fixtures under ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    for path in build_tiny_fixtures(args.out):
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
