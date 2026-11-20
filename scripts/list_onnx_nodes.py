#!/usr/bin/env python
"""Print the node names and operator types of an ONNX graph.

Used once before the first real export (T12) to confirm that the ``mlp_only``
exclusion patterns of ``stc.adapters.export.quantization`` match the actual
node names produced by the MrBERT-es export.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Print ``op_type<TAB>name`` for every node, then exit 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ONNX model file")
    args = parser.parse_args(argv)
    try:
        import onnx
    except ImportError:
        print("onnx is required: install the 'train' extra")
        return 2
    model = onnx.load(str(args.model))
    for node in model.graph.node:
        print(f"{node.op_type}\t{node.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
