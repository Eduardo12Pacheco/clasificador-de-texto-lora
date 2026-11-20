"""The shared tokenizer adapter truncates, pads and preserves batch order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stc.adapters.text.hf_tokenizer import HfTokenizer, resolve_tokenizer_file
from stc.domain.errors import InvalidInputError

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tiny" / "tokenizer.json"


def _tokenizer(**kwargs: object) -> HfTokenizer:
    return HfTokenizer.from_file(TINY_TOKENIZER, **kwargs)


def test_pad_and_unknown_ids_come_from_the_tokenizer_file() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.pad_id == 0
    assert tokenizer.vocab_size == 1148
    assert tokenizer.max_length == 64


def test_batches_are_dynamically_padded_and_ordered() -> None:
    tokenizer = _tokenizer()
    texts = ["prende la luz", "ponme música de Juanes"]

    encoded = tokenizer.encode_batch(texts)

    assert encoded.input_ids.shape == (2, 4)
    assert encoded.attention_mask.shape == (2, 4)
    assert encoded.attention_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert encoded.input_ids[0, 0] == tokenizer.encode_batch(["prende"]).input_ids[0, 0]
    assert np.all(encoded.input_ids[0, 3] == tokenizer.pad_id)


def test_truncation_respects_max_length() -> None:
    tokenizer = _tokenizer(max_length=3)

    encoded = tokenizer.encode_batch(["una frase mucho más larga que tres tokens"])

    assert encoded.input_ids.shape == (1, 3)
    assert encoded.attention_mask.sum() == 3


def test_empty_batch_has_a_valid_empty_shape() -> None:
    encoded = _tokenizer().encode_batch([])

    assert encoded.input_ids.shape == (0, 0)
    assert encoded.attention_mask.shape == (0, 0)


def test_missing_tokenizer_file_is_an_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="tokenizer file not found"):
        HfTokenizer.from_file(tmp_path / "absent.json")


def test_tiny_backbone_requires_an_explicit_tokenizer_path() -> None:
    with pytest.raises(InvalidInputError, match="explicit tokenizer_path"):
        resolve_tokenizer_file("stc-tiny-test", None)


def test_explicit_tokenizer_path_is_returned() -> None:
    assert resolve_tokenizer_file("anything", TINY_TOKENIZER) == TINY_TOKENIZER
