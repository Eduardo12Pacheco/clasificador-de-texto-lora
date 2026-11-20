"""Tests for the pure text normalisation helpers of the domain layer."""

import re
import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stc.domain.text import normalize_for_dedup, strip_accents

pytestmark = pytest.mark.unit

COMBINING_MARK = re.compile(r"[\u0300-\u036f]")


def test_normalize_for_dedup_strips_case_punctuation_and_accents() -> None:
    assert (
        normalize_for_dedup("¿Despiértame  a las CINCO, de la mañana!")
        == "despiertame a las cinco de la manana"
    )


def test_normalize_for_dedup_collapses_whitespace_and_strips() -> None:
    assert normalize_for_dedup("  pon\tmúsica \n ahora  ") == "pon musica ahora"


def test_normalize_for_dedup_keeps_word_characters() -> None:
    assert normalize_for_dedup("intent_1 sub_score-2") == "intent_1 sub_score2"


@given(st.text())
def test_normalize_for_dedup_is_idempotent(text: str) -> None:
    once = normalize_for_dedup(text)
    assert normalize_for_dedup(once) == once


@given(st.text())
def test_normalize_for_dedup_output_is_clean(text: str) -> None:
    result = normalize_for_dedup(text)
    assert not re.search(r"[^\w\s]", result)
    assert not re.search(r"\s{2,}", result)
    assert result == result.strip()
    assert not any(unicodedata.category(char).startswith("M") for char in result)


def test_strip_accents_removes_marks() -> None:
    assert strip_accents("Despiértame mañana") == "Despiertame manana"


def test_strip_accents_keeps_tilde_on_n_when_requested() -> None:
    assert strip_accents("Despiértame mañana", keep_tilde_n=True) == "Despiertame mañana"
    assert strip_accents("AÑO", keep_tilde_n=True) == "AÑO"


def test_strip_accents_is_idempotent() -> None:
    text = "Canción de la niña, ¿dónde está?"
    assert strip_accents(strip_accents(text)) == strip_accents(text)


@pytest.mark.parametrize("text", ["", " ", "Ñandú", "ÁRBOL", "café"])
def test_strip_accents_leaves_no_combining_marks(text: str) -> None:
    assert not COMBINING_MARK.search(strip_accents(text))
