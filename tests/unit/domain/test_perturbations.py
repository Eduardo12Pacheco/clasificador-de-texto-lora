"""Orthographic and dialect perturbations of decision D11 (S9)."""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stc.domain.perturbations import (
    PERTURBATIONS,
    chat_abbrev,
    no_accents,
    regionalize,
    typos,
    uppercase,
)

pytestmark = pytest.mark.unit

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for row, char_a in enumerate(a, start=1):
        current = [row]
        for column, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def test_perturbation_registry_names_the_seven_variants() -> None:
    assert set(PERTURBATIONS) == {
        "no_accents",
        "uppercase",
        "typos",
        "chat_abbrev",
        "regionalize:mx",
        "regionalize:ec",
        "regionalize:ar",
    }


def test_no_accents_removes_diacritics_but_keeps_tilde_n() -> None:
    assert no_accents("Despiértame mañana, por favor") == "Despiertame mañana, por favor"
    assert no_accents("pingüino") == "pinguino"


def test_uppercase_is_a_plain_upper_case() -> None:
    assert uppercase("Ponme música") == "PONME MÚSICA"


def test_chat_abbrev_replaces_only_whole_words() -> None:
    assert chat_abbrev("por favor, que hora es") == "xfa, q hora es"
    assert chat_abbrev("quiero que vengas") == "quiero q vengas"
    assert chat_abbrev("porque quiero") == "xq quiero"


def test_typos_is_deterministic_and_seed_sensitive() -> None:
    text = "Despiértame mañana temprano porque quiero música y luego noticias del tiempo"

    assert typos(text, seed=13, rate=0.5) == typos(text, seed=13, rate=0.5)
    assert typos(text, seed=13, rate=0.5) != typos(text, seed=42, rate=0.5)


def test_typos_edits_each_selected_word_by_at_most_one_character() -> None:
    text = "Despiértame mañana temprano porque quiero música y luego noticias del tiempo"
    edited = typos(text, seed=13, rate=1.0)
    original_words = _words(text)
    edited_words = _words(edited)

    assert len(original_words) == len(edited_words)
    assert any(a != b for a, b in zip(original_words, edited_words, strict=True))
    assert all(_levenshtein(a, b) <= 1 for a, b in zip(original_words, edited_words, strict=True))
    assert "y" in edited.split()  # short words are never edited


def test_typos_preserves_punctuation_and_length() -> None:
    text = "¿Despiértame a las seis, por favor?"

    edited = typos(text, seed=13, rate=1.0)

    assert len(edited) >= len(text) - 3
    assert edited.startswith("¿")
    assert edited.endswith("?")


def test_regionalize_profile_examples() -> None:
    assert regionalize("Ponme música ahora mismo", "ar") == "Poneme música ahora mismo"
    assert regionalize("Ponme música ahora mismo", "mx") == "Ponme música ahorita"
    assert regionalize("Apaga el enchufe", "ec") == "Apaga el tomacorriente"
    assert regionalize("Llévame en coche", "ar") == "Llévame en auto"
    assert regionalize("Pon el mando a distancia", "mx") == "Pon el control remoto"


@given(
    st.lists(
        st.sampled_from(["móvil", "coche", "fresa", "autobús", "penal", "luz"]),
        max_size=10,
    )
)
def test_regionalize_keeps_digits_and_punctuation(words: list[str]) -> None:
    decorated = f"¡{' '.join(words)} 3!"
    result = regionalize(decorated, "ec")
    assert result.startswith("¡")
    assert result.endswith(" 3!")
