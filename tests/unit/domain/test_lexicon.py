"""LatAm lexicon tables and whole-word, case-insensitive substitution."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stc.domain.errors import InvalidInputError
from stc.domain.lexicon import (
    CHAT_ABBREV,
    COMMON_LATAM,
    PROFILES,
    replace_terms,
)

pytestmark = pytest.mark.unit


def test_profiles_share_common_latam_and_are_lowercase() -> None:
    assert set(PROFILES) == {"mx", "ec", "ar"}
    for profile, table in PROFILES.items():
        assert profile in {"mx", "ec", "ar"}
        assert all(key == key.lower() for key in table), profile
        assert all(key == key.lower() for key in COMMON_LATAM)
        assert set(COMMON_LATAM) <= set(table)


def test_no_replacement_value_is_also_a_key() -> None:
    for name, table in {"common": COMMON_LATAM, "chat": CHAT_ABBREV, **PROFILES}.items():
        keys = set(table)
        assert not (set(table.values()) & keys), name


def test_matching_is_whole_word_case_insensitive_and_longest_first() -> None:
    assert replace_terms("Coche, coches y coche.", {"coche": "auto"}) == "Auto, coches y auto."
    assert (
        replace_terms("Ahora mismo quiero ir.", {"ahora mismo": "ahorita", "ahora": "ahora"})
        == "Ahorita quiero ir."
    )
    assert replace_terms("El quesito, por favor.", {"que": "q"}) == "El quesito, por favor."


def test_punctuation_and_digits_are_preserved() -> None:
    text = "¿Dónde está el autobús? Llega a las 3, ¡por favor!"

    assert replace_terms(text, {"autobús": "bus", "por favor": "xfa"}) == (
        "¿Dónde está el bus? Llega a las 3, ¡xfa!"
    )


def test_capital_letter_is_preserved_in_replacement() -> None:
    assert replace_terms("Enciende la luz", {"enciende": "prende"}) == "Prende la luz"
    assert replace_terms("Enciéndeme la luz", {"enciéndeme": "préndeme"}) == "Préndeme la luz"


def test_rejects_unknown_profile() -> None:
    from stc.domain.perturbations import regionalize

    with pytest.raises(InvalidInputError):
        regionalize("texto", "cl")


@given(
    st.lists(
        st.sampled_from(["hola", "música", "quiero", "tiempo", "casa", "agua", "luz", "correo"]),
        max_size=8,
    )
)
def test_text_outside_the_lexicon_is_unchanged(words: list[str]) -> None:
    from stc.domain.perturbations import regionalize

    text = " ".join(words)
    assert regionalize(text, "ar") == text


@given(
    st.lists(
        st.sampled_from(["coche", "móvil", "ahora", "enciende", "autobús", "fresa", "habitación"]),
        max_size=8,
    )
)
def test_regionalize_is_idempotent(words: list[str]) -> None:
    from stc.domain.perturbations import regionalize

    text = " ".join(words)
    once = regionalize(text, "mx")
    assert regionalize(once, "mx") == once
