"""The pre-registered D6 backbone-selection rule (pure)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stc.application.dto import ScreeningRun
from stc.domain.errors import InvalidInputError

__all__ = ["SELECTION_RULE", "BackboneSelection", "select_backbone"]

SELECTION_RULE = (
    "among exportable candidates choose the highest mean dev macro-F1; "
    "MrBERT-es wins when within 1.0 point of bertin; bertin wins only when "
    "it is more than 1.0 point ahead"
)
MRBERT = "BSC-LT/MrBERT-es"
BERTIN = "bertin-project/bertin-roberta-base-spanish"
QWEN = "Qwen/Qwen3-0.6B"
_ONE_POINT = 0.01


@dataclass(frozen=True, slots=True)
class BackboneSelection:
    """The chosen backbone and the branch of the D6 rule that fired."""

    selected: str
    rule_branch: str
    per_backbone_means: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable selection."""
        return {
            "schema_version": 1,
            "selected": self.selected,
            "rule_branch": self.rule_branch,
            "rule": SELECTION_RULE,
            "per_backbone_means": dict(self.per_backbone_means),
        }


def _mean(runs: Sequence[ScreeningRun], backbone: str) -> float:
    values = [run.dev_macro_f1 for run in runs if run.backbone == backbone]
    if not values:
        raise InvalidInputError(f"screening has no runs for {backbone}")
    return float(np.mean(values))


def select_backbone(runs: Sequence[ScreeningRun]) -> BackboneSelection:
    """Apply D6; Qwen is never selectable even when it scores best."""
    if not runs:
        raise InvalidInputError("screening produced no runs")
    means = {backbone: _mean(runs, backbone) for backbone in sorted({run.backbone for run in runs})}
    mrbert = means.get(MRBERT)
    bertin = means.get(BERTIN)
    if mrbert is None:
        if bertin is None:
            raise InvalidInputError("screening has neither MrBERT-es nor bertin")
        return BackboneSelection(BERTIN, "bertin_ahead", means)
    if bertin is None:
        return BackboneSelection(MRBERT, "mrbert_ahead", means)
    if bertin - mrbert > _ONE_POINT:
        return BackboneSelection(BERTIN, "bertin_ahead", means)
    if mrbert >= bertin:
        return BackboneSelection(MRBERT, "mrbert_ahead", means)
    return BackboneSelection(MRBERT, "within_1pt_prefer_mrbert", means)
