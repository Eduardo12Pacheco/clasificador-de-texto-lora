"""Rust ``tokenizers`` adapter shared by training and CPU inference (D8)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from stc.application.dto import EncodedBatch
from stc.domain.errors import InvalidInputError

__all__ = ["HfTokenizer", "resolve_tokenizer_file"]

# Encoder tokenizers use ``<pad>`` (MrBERT-es id 1, RoBERTa id 1). Decoder
# tokenizers such as Qwen3 have no ``<pad>`` entry; their canonical padding
# token is ``<|endoftext|>``. The remaining candidates cover other common
# special-token vocabularies without changing encoder behaviour.
_PAD_TOKEN_FALLBACKS = ("<|endoftext|>", "</s>", "<unk>")


class HfTokenizer:
    """Encode text with a backbone ``tokenizer.json`` and dynamic padding.

    Training and inference share this class so there is no train/serve skew:
    truncation happens on the right at ``max_length`` and padding only extends
    to the longest sequence of the current batch.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_length: int = 64,
        pad_token: str = "<pad>",  # noqa: S107 - a vocabulary token, not a secret
    ) -> None:
        if max_length < 1:
            raise InvalidInputError("max_length must be positive")
        self._tokenizer = tokenizer
        self._tokenizer.enable_truncation(max_length=max_length)
        self._max_length = max_length
        token_id = _resolve_pad_id(tokenizer, pad_token)
        self._pad_id: int = token_id

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        max_length: int = 64,
        pad_token: str = "<pad>",  # noqa: S107 - a vocabulary token, not a secret
    ) -> HfTokenizer:
        """Load a tokenizer JSON file from disk."""
        source = Path(path)
        if not source.is_file():
            raise InvalidInputError(f"tokenizer file not found: {source}")
        try:
            tokenizer = Tokenizer.from_file(str(source))
        except Exception as error:  # tokenizers raises its own Exception subclass
            raise InvalidInputError(f"cannot load tokenizer {source}: {error}") from error
        return cls(tokenizer, max_length=max_length, pad_token=pad_token)

    @property
    def max_length(self) -> int:
        """Maximum sequence length in tokens."""
        return self._max_length

    @property
    def pad_id(self) -> int:
        """Id of the padding token."""
        return self._pad_id

    @property
    def vocab_size(self) -> int:
        """Number of tokens known to the tokenizer."""
        return int(self._tokenizer.get_vocab_size())

    def encode_batch(self, texts: Sequence[str]) -> EncodedBatch:
        """Encode texts into right-truncated, dynamically padded id arrays."""
        items = [str(text) for text in texts]
        if not items:
            empty = np.zeros((0, 0), dtype=np.int64)
            return EncodedBatch(input_ids=empty, attention_mask=empty.copy())
        encodings = self._tokenizer.encode_batch(items)
        width = max(1, max(len(encoding.ids) for encoding in encodings))
        input_ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            if length:
                input_ids[row, :length] = _as_int64(encoding.ids)
                attention_mask[row, :length] = 1
        return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask)


def _resolve_pad_id(tokenizer: Tokenizer, preferred: str) -> int:
    """Return ``preferred`` or the first known special-token fallback."""
    for candidate in (preferred, *_PAD_TOKEN_FALLBACKS):
        token_id = tokenizer.token_to_id(candidate)
        if token_id is not None:
            return int(token_id)
    known = ", ".join(repr(name) for name in (preferred, *_PAD_TOKEN_FALLBACKS))
    raise InvalidInputError(f"tokenizer has none of the padding tokens: {known}")


def _as_int64(values: Sequence[int]) -> NDArray[np.int64]:
    return np.asarray(values, dtype=np.int64)


def resolve_tokenizer_file(backbone: str, explicit: str | Path | None) -> Path:
    """Resolve an explicit tokenizer file or download ``tokenizer.json``.

    Tests and the ``stc-tiny-test`` backbone always provide an explicit path;
    real backbones are resolved through the Hugging Face cache, which is only
    touched by the GPU training task.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise InvalidInputError(f"tokenizer file not found: {path}")
        return path
    if backbone == "stc-tiny-test":
        raise InvalidInputError("stc-tiny-test needs an explicit tokenizer_path in its config")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise InvalidInputError(
            "downloading a tokenizer needs huggingface_hub; install the 'train' extra"
        ) from error
    return Path(hf_hub_download(repo_id=backbone, filename="tokenizer.json"))
