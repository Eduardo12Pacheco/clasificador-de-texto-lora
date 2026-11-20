#!/usr/bin/env python
"""Build the deterministic tiny artifacts used by the tests.

Two artifacts are generated:

* ``tests/fixtures/tiny/tokenizer.json`` (T7): a WordLevel tokenizer whose
  vocabulary comes from the bundled MASSIVE sample, used by the torch tests.
* ``tests/fixtures/tiny_bundle/`` (T8): a self-contained ONNX bundle (a
  bag-of-embeddings graph, tokenizer, labels, calibration and a manifest with
  real sha256s) used by the inference, hub, pack and CLI tests.

Everything is generated from committed inputs, so running the script twice
produces byte-identical files. The ONNX bundle needs the ``train`` extra
(``onnx``); without it the script still writes the tokenizer and says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "src" / "stc" / "_fixture" / "massive_es_sample"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "tiny"
DEFAULT_BUNDLE_OUT = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"

PAD_TEXT = "<pad>"
UNK_TEXT = "<unk>"

BUNDLE_NAME = "stc-tiny-test"
BUNDLE_VERSION = "0.0.1"
BUNDLE_CREATED_AT = "2026-11-20T00:00:00+00:00"
BUNDLE_HIDDEN = 16
BUNDLE_SOURCE_ROWS = 200
BUNDLE_SEED = 13
BUNDLE_TEMPERATURE = 1.5
BUNDLE_REVIEW_THRESHOLD = 0.5


def _fixture_rows(split: str) -> list[dict[str, object]]:
    path = FIXTURE_DIR / f"{split}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _fixture_texts() -> list[str]:
    return [str(row["text"]) for split in ("train", "dev", "test") for row in _fixture_rows(split)]


def vocabulary_from_texts(texts: Iterable[str]) -> dict[str, int]:
    """Build the sorted WordLevel vocabulary with the two special tokens first."""
    words = sorted({word for text in texts for word in text.split()})
    vocab = {PAD_TEXT: 0, UNK_TEXT: 1}
    for index, word in enumerate(words, start=2):
        vocab[word] = index
    return vocab


def build_tokenizer_payload(texts: Iterable[str]) -> dict[str, object]:
    """Return the tokenizers JSON document for a tiny WordLevel tokenizer."""
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
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tiny_fixtures(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    """Generate the T7 tokenizer fixture and return the paths written."""
    payload = build_tokenizer_payload(_fixture_texts())
    tokenizer_path = Path(out_dir) / "tokenizer.json"
    write_json(tokenizer_path, payload)
    return [tokenizer_path]


def _bundle_onnx_model(vocab_size: int, n_labels: int) -> bytes:
    """Build the bag-of-embeddings ONNX graph described in T8."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(BUNDLE_SEED)
    embedding = rng.normal(0.0, 0.1, size=(vocab_size, BUNDLE_HIDDEN)).astype(np.float32)
    projection = rng.normal(0.0, 0.1, size=(BUNDLE_HIDDEN, n_labels)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(n_labels,)).astype(np.float32)

    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["embedded"], axis=0),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_float", "unsqueeze_axes"], ["mask_3d"]),
        helper.make_node("Mul", ["embedded", "mask_3d"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "reduce_axes"], ["pooled"], keepdims=0),
        helper.make_node("MatMul", ["pooled", "projection"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny-bag-of-embeddings",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_labels])],
        initializer=[
            numpy_helper.from_array(embedding, "embedding"),
            numpy_helper.from_array(projection, "projection"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), "unsqueeze_axes"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "reduce_axes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString(deterministic=True))


def build_tiny_bundle(bundle_dir: Path = DEFAULT_BUNDLE_OUT) -> list[Path]:
    """Generate the T8 ONNX bundle and return the paths written."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("skipping tiny ONNX bundle: install the 'train' extra to build it")
        return []

    from stc import __version__

    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = _fixture_rows("train")[:BUNDLE_SOURCE_ROWS]
    texts = [str(row["text"]) for row in rows]
    tokenizer_payload = build_tokenizer_payload(texts)
    write_json(root / "tokenizer.json", tokenizer_payload)
    vocab_size = len(tokenizer_payload["model"]["vocab"])  # type: ignore[index]

    intents = sorted(
        {str(row["intent"]) for split in ("train", "dev", "test") for row in _fixture_rows(split)}
    )
    write_json(root / "labels.json", {"schema_version": 1, "intents": intents})
    (root / "model.onnx").write_bytes(_bundle_onnx_model(vocab_size, len(intents)))
    write_json(
        root / "calibration.json",
        {
            "schema_version": 1,
            "temperature": BUNDLE_TEMPERATURE,
            "review_threshold": BUNDLE_REVIEW_THRESHOLD,
        },
    )

    content_files = ("model.onnx", "tokenizer.json", "labels.json", "calibration.json")
    manifest = {
        "bundle_version": 1,
        "name": BUNDLE_NAME,
        "version": BUNDLE_VERSION,
        "kind": "onnx-seqcls",
        "created_at": BUNDLE_CREATED_AT,
        "stc_version": __version__,
        "model": {
            "parameters": vocab_size * BUNDLE_HIDDEN + BUNDLE_HIDDEN * len(intents) + len(intents),
            "trainable_parameters": 0,
        },
        "base_model": {"id": BUNDLE_NAME, "revision": "fixture", "license": "Apache-2.0"},
        "adapter": {"method": "none"},
        "dataset": {"id": "massive-es-fixture", "tarball_sha256": "fixture"},
        "training_run_id": "fixture",
        "quantization": {"mode": "fp32"},
        "onnx": {
            "file": "model.onnx",
            "opset": 17,
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["logits"],
        },
        "tokenizer": {"max_length": 64, "pad_id": 0},
        "files": [
            {
                "path": name,
                "sha256": _sha256(root / name),
                "size": (root / name).stat().st_size,
            }
            for name in content_files
        ],
    }
    write_json(root / "manifest.json", manifest)
    return [root / name for name in (*content_files, "manifest.json")]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the tiny fixtures under ``--out`` and the bundle under ``--bundle-out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bundle-out", type=Path, default=DEFAULT_BUNDLE_OUT)
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Skip the ONNX bundle (used when the train extra is not installed).",
    )
    args = parser.parse_args(argv)
    written = build_tiny_fixtures(args.out)
    if not args.tokenizer_only:
        written.extend(build_tiny_bundle(args.bundle_out))
    for path in written:
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
