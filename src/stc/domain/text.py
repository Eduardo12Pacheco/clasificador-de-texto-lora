"""Pure text normalisation helpers shared by leakage checks and adapters."""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_TILDE = "\u0303"


def _is_mark(character: str) -> bool:
    return unicodedata.category(character).startswith("M")


def strip_accents(text: str, *, keep_tilde_n: bool = False) -> str:
    """Remove diacritics, optionally preserving the tilde on ``n``/``N``.

    ``keep_tilde_n`` is used by the ``no_accents`` robustness perturbation so
    that Spanish ``ñ`` stays a distinct letter.
    """
    decomposed = unicodedata.normalize("NFD", text)
    output: list[str] = []
    for character in decomposed:
        if not _is_mark(character):
            output.append(character)
            continue
        if keep_tilde_n and character == _TILDE and output and output[-1] in {"n", "N"}:
            output.append(character)
    return unicodedata.normalize("NFC", "".join(output))


def normalize_for_dedup(text: str) -> str:
    """Normalise text exactly as the leakage policy (decision D3) defines.

    casefold -> Unicode NFKD -> drop combining marks -> remove every character
    that is not a word character or whitespace -> collapse whitespace -> strip.

    A final casefold keeps the function idempotent: NFKD compatibility
    decompositions can reintroduce uppercase code points (for example
    ``U+A7F2``) after the initial casefold, so the output is casefolded once
    more before whitespace is collapsed.
    """
    folded = text.casefold()
    decomposed = unicodedata.normalize("NFKD", folded)
    without_marks = "".join(character for character in decomposed if not _is_mark(character))
    without_punctuation = _PUNCTUATION.sub("", without_marks).casefold()
    return _WHITESPACE.sub(" ", without_punctuation).strip()
