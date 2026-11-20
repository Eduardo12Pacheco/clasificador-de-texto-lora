"""Label space: intents, their derived scenarios and index mappings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import LabelSpaceError

_INTENT_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")


@dataclass(frozen=True, slots=True)
class LabelSpace:
    """A validated, sorted set of intents with their derived scenarios."""

    intents: tuple[str, ...]
    scenarios: tuple[str, ...]
    _scenario_map: NDArray[np.int64] = field(repr=False, compare=False)

    @classmethod
    def from_intents(cls, intents: Iterable[str]) -> LabelSpace:
        """Build a label space, sorting intents and deriving scenarios."""
        materialized = tuple(intents)
        if not materialized:
            raise LabelSpaceError("label space must contain at least one intent")
        for intent in materialized:
            if not isinstance(intent, str) or _INTENT_NAME.fullmatch(intent) is None:
                raise LabelSpaceError(f"invalid intent name: {intent!r}")
        if len(set(materialized)) != len(materialized):
            raise LabelSpaceError("duplicate intent names are not allowed")

        sorted_intents = tuple(sorted(materialized))
        scenarios = tuple(sorted({intent.split("_", maxsplit=1)[0] for intent in sorted_intents}))
        scenario_index = {scenario: index for index, scenario in enumerate(scenarios)}
        scenario_map = np.array(
            [scenario_index[intent.split("_", maxsplit=1)[0]] for intent in sorted_intents],
            dtype=np.int64,
        )
        scenario_map.flags.writeable = False
        return cls(intents=sorted_intents, scenarios=scenarios, _scenario_map=scenario_map)

    @property
    def n_intents(self) -> int:
        """Number of intents in the space."""
        return len(self.intents)

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios in the space."""
        return len(self.scenarios)

    @property
    def scenario_map(self) -> NDArray[np.int64]:
        """Read-only array mapping each intent index to its scenario index."""
        return self._scenario_map

    def intent_index(self, name: str) -> int:
        """Return the index of ``name`` or raise :class:`LabelSpaceError`."""
        try:
            return self.intents.index(name)
        except ValueError as error:
            raise LabelSpaceError(f"unknown intent: {name!r}") from error

    def scenario_index(self, name: str) -> int:
        """Return the index of ``name`` or raise :class:`LabelSpaceError`."""
        try:
            return self.scenarios.index(name)
        except ValueError as error:
            raise LabelSpaceError(f"unknown scenario: {name!r}") from error

    def scenario_of_intent(self, intent_index: int) -> int:
        """Return the scenario index of an intent index."""
        if not 0 <= intent_index < self.n_intents:
            raise LabelSpaceError(f"intent index out of range: {intent_index}")
        return int(self._scenario_map[intent_index])

    def to_scenario_indices(self, intent_indices: NDArray[np.int64]) -> NDArray[np.int64]:
        """Map an array of intent indices to scenario indices."""
        indices = np.asarray(intent_indices)
        if indices.ndim != 1:
            raise LabelSpaceError("intent indices must be a one-dimensional array")
        if not np.issubdtype(indices.dtype, np.integer):
            raise LabelSpaceError("intent indices must be integers")
        if np.any(indices < 0) or np.any(indices >= self.n_intents):
            raise LabelSpaceError("intent index out of range")
        return np.asarray(self._scenario_map[indices], dtype=np.int64)
