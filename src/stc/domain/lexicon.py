"""Whole-word Latin-American Spanish lexicon used by the robustness probe.

The substitutions are deliberately conservative: every key and value keep the
same gender and number, so a sentence remains grammatical after regionalisation.
``COMMON_LATAM`` holds the pan-regional changes and each profile overlays its
own table on top of it. The tables are data, not behaviour: the matching rules
live in :func:`replace_terms` and are shared with the chat-abbreviation
perturbation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache

from stc.domain.errors import InvalidInputError

__all__ = [
    "CHAT_ABBREV",
    "COMMON_LATAM",
    "PROFILES",
    "REGIONAL_PROFILES",
    "replace_terms",
]

COMMON_LATAM: dict[str, str] = {
    "móvil": "celular",
    "móviles": "celulares",
    "zumo": "jugo",
    "zumos": "jugos",
    "patata": "papa",
    "patatas": "papas",
    "aparcar": "estacionar",
    "aparca": "estaciona",
    "aparcamiento": "estacionamiento",
    "conducir": "manejar",
    "melocotón": "durazno",
    "melocotones": "duraznos",
    "mando a distancia": "control remoto",
    "bocadillo": "sándwich",
    "bocadillos": "sándwiches",
    "coche": "carro",
    "coches": "carros",
    "billete": "boleto",
    "billetes": "boletos",
    "jersey": "suéter",
    "jerséis": "suéteres",
}

_MX: dict[str, str] = {
    "ahora mismo": "ahorita",
    "ahora": "ahorita",
    "enciende": "prende",
    "encender": "prender",
    "enciéndeme": "préndeme",
    "autobús": "camión",
    "autobuses": "camiones",
    "guisantes": "chícharos",
    "enchufe": "contacto",
    "enchufes": "contactos",
    "cacahuete": "cacahuate",
    "cacahuetes": "cacahuates",
}

_EC: dict[str, str] = {
    "ahora mismo": "ahorita",
    "ahora": "ahorita",
    "enciende": "prende",
    "encender": "prender",
    "enciéndeme": "préndeme",
    "billete": "pasaje",
    "billetes": "pasajes",
    "autobús": "bus",
    "autobuses": "buses",
    "nevera": "refrigeradora",
    "fresa": "frutilla",
    "fresas": "frutillas",
    "judías verdes": "vainitas",
    "tarta": "torta",
    "tartas": "tortas",
    "bolígrafo": "esfero",
    "bolígrafos": "esferos",
    "chaqueta": "chompa",
    "chaquetas": "chompas",
    "enchufe": "tomacorriente",
    "enchufes": "tomacorrientes",
    "cacahuete": "maní",
    "cacahuetes": "maníes",
}

_AR: dict[str, str] = {
    "coche": "auto",
    "coches": "autos",
    "autobús": "colectivo",
    "autobuses": "colectivos",
    "nevera": "heladera",
    "fresa": "frutilla",
    "fresas": "frutillas",
    "judías verdes": "chauchas",
    "tarta": "torta",
    "tartas": "tortas",
    "chaqueta": "campera",
    "chaquetas": "camperas",
    "jersey": "pulóver",
    "jerséis": "pulóveres",
    "habitación": "pieza",
    "habitaciones": "piezas",
    "cacahuete": "maní",
    "cacahuetes": "maníes",
    # Rioplatense voseo imperatives.
    "enciende": "encendé",
    "pon": "poné",
    "ponme": "poneme",
    "apaga": "apagá",
    "dime": "decime",
    "busca": "buscá",
    "reproduce": "reproducí",
    "abre": "abrí",
    "cierra": "cerrá",
    "recuérdame": "recordame",
    "avísame": "avisame",
    "despiértame": "despertame",
    "muéstrame": "mostrame",
    "llama": "llamá",
    "envía": "enviá",
    "manda": "mandá",
    "cambia": "cambiá",
    "silencia": "silenciá",
    "activa": "activá",
    "desactiva": "desactivá",
    "borra": "borrá",
    "añade": "agregá",
}

REGIONAL_PROFILES: dict[str, dict[str, str]] = {"mx": _MX, "ec": _EC, "ar": _AR}

PROFILES: dict[str, dict[str, str]] = {
    profile: {**COMMON_LATAM, **specific} for profile, specific in REGIONAL_PROFILES.items()
}

CHAT_ABBREV: dict[str, str] = {
    "por favor": "xfa",
    "por qué": "xq",
    "porque": "xq",
    "que": "q",
    "qué": "q",
    "también": "tb",
    "tambien": "tb",
    "para": "pa",
    "gracias": "grax",
    "mensaje": "msj",
    "mensajes": "msjs",
    "cumpleaños": "cumple",
}


@lru_cache(maxsize=32)
def _compiled(
    items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], dict[str, str]]:
    keys = sorted((key for key, _ in items), key=lambda key: (-len(key), key))
    alternatives = "|".join(f"(?<!\\w){re.escape(key)}(?!\\w)" for key in keys)
    return re.compile(alternatives, re.IGNORECASE), {key.casefold(): value for key, value in items}


def replace_terms(text: str, mapping: Mapping[str, str]) -> str:
    """Replace whole words of ``mapping``, longest key first, preserving capitals.

    Matching is case-insensitive. If the matched source starts with an upper-case
    letter, the replacement starts with one too (``Enciende`` -> ``Prende``).
    Digits and punctuation are never touched.
    """
    if not mapping:
        return text
    items = tuple(sorted(mapping.items()))
    pattern, lookup = _compiled(items)
    if not items:
        return text

    def substitution(match: re.Match[str]) -> str:
        source = match.group(0)
        replacement = lookup[source.casefold()]
        if source[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(substitution, text)


def profile_table(profile: str) -> dict[str, str]:
    """Return the table for ``profile`` or raise a domain error."""
    try:
        return PROFILES[profile]
    except KeyError as error:
        raise InvalidInputError(
            f"unknown regionalisation profile: {profile!r}; expected one of "
            f"{', '.join(sorted(PROFILES))}"
        ) from error
