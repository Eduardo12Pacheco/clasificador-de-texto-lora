"""Turn calibrated predictions into the stable ``stc predict`` payload (S13)."""

from __future__ import annotations

from collections.abc import Sequence

from stc.application.services.calibrated_classifier import CalibratedClassifier
from stc.domain.errors import InvalidInputError

__all__ = ["PredictTexts"]


class PredictTexts:
    """Validate texts and build the versioned prediction payload shared by CLI/API."""

    def run(
        self,
        classifier: CalibratedClassifier,
        texts: Sequence[str],
        *,
        top_k: int = 3,
    ) -> dict[str, object]:
        """Return the schema-versioned prediction document for ``texts``."""
        items = [str(text).strip() for text in texts]
        if not items or any(not item for item in items):
            raise InvalidInputError("prediction needs at least one non-empty text")
        predictions = classifier.predict(items, top_k=top_k)
        info = classifier.info
        return {
            "schema_version": 1,
            "model": {
                "id": info.model_id,
                "kind": info.kind,
                "version": info.version,
                "quantization": info.quantization,
                "temperature": classifier.calibration.temperature,
                "review_threshold": classifier.calibration.review_threshold,
            },
            "predictions": [
                {
                    "index": index,
                    "intent": prediction.intent,
                    "scenario": prediction.scenario,
                    "confidence": prediction.confidence,
                    "needs_review": prediction.needs_review,
                    "top_k": [
                        {
                            "intent": label.intent,
                            "scenario": label.scenario,
                            "probability": label.probability,
                        }
                        for label in prediction.top_k
                    ],
                }
                for index, prediction in enumerate(predictions)
            ],
        }
