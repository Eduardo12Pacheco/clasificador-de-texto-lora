"""Perturbation functions of decision D11, used by the S9 robustness suite."""

from __future__ import annotations

import re
from collections.abc import Callable
from random import Random

from stc.domain.errors import InvalidInputError
from stc.domain.lexicon import CHAT_ABBREV, profile_table, replace_terms
from stc.domain.text import strip_accents

__all__ = [
    "PERTURBATIONS",
    "apply_perturbation",
    "chat_abbrev",
    "no_accents",
    "regionalize",
    "typos",
    "uppercase",
]

_ALPHABETIC_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def no_accents(text: str) -> str:
    """Remove diacritics while keeping ``ñ`` distinct (S9)."""
    return strip_accents(text, keep_tilde_n=True)


def uppercase(text: str) -> str:
    """Upper-case the whole utterance."""
    return text.upper()


_SUBSTITUTES = "abcdefghijklmnopqrstuvwxyzáéíóúñ"


def _mutate(word: str, rng: Random) -> str:
    # D11 lists "swap adjacent letters"; an adjacent transposition has
    # Levenshtein distance 2, while the S9 acceptance test pins every edit to
    # distance <= 1. A one-character substitution keeps that invariant and the
    # same realistic "fat-finger" effect, so it replaces the transposition.
    operation = rng.randrange(3)
    if operation == 0:
        index = rng.randrange(len(word))
        choices = [char for char in _SUBSTITUTES if char != word[index]]
        replacement = choices[rng.randrange(len(choices))]
        return word[:index] + replacement + word[index + 1 :]
    if operation == 1:
        index = rng.randrange(len(word))
        return word[:index] + word[index + 1 :]
    index = rng.randrange(len(word))
    return word[: index + 1] + word[index] + word[index + 1 :]


def typos(text: str, seed: int = 13, rate: float = 0.06) -> str:
    """Apply one character-level typo to each selected long word (D11).

    The random generator is seeded with ``f"{seed}:{text}"``, so the same text
    and seed always produce the same output, independently of call order.
    """
    if not 0.0 <= rate <= 1.0:
        raise InvalidInputError("typo rate must lie in [0, 1]")
    # Reproducible data augmentation, not cryptography.
    rng = Random(f"{seed}:{text}")  # noqa: S311

    def edit(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) < 4 or rng.random() >= rate:
            return word
        return _mutate(word, rng)

    return _ALPHABETIC_WORD.sub(edit, text)


def chat_abbrev(text: str) -> str:
    """Replace words and phrases with their chat abbreviations (D11)."""
    return replace_terms(text, CHAT_ABBREV)


def regionalize(text: str, profile: str) -> str:
    """Apply the whole-word dialect lexicon for ``mx``, ``ec`` or ``ar`` (D11)."""
    return replace_terms(text, profile_table(profile))


def _mx(text: str) -> str:
    return regionalize(text, "mx")


def _ec(text: str) -> str:
    return regionalize(text, "ec")


def _ar(text: str) -> str:
    return regionalize(text, "ar")


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "no_accents": no_accents,
    "uppercase": uppercase,
    "typos": typos,
    "chat_abbrev": chat_abbrev,
    "regionalize:mx": _mx,
    "regionalize:ec": _ec,
    "regionalize:ar": _ar,
}


def apply_perturbation(name: str, text: str) -> str:
    """Apply the registered perturbation ``name`` to one utterance."""
    try:
        function = PERTURBATIONS[name]
    except KeyError as error:
        raise InvalidInputError(
            f"unknown perturbation: {name!r}; expected one of {', '.join(PERTURBATIONS)}"
        ) from error
    return function(text)
