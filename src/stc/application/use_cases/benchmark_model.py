"""CPU latency benchmark use case (S12)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from stc.application.ports import Classifier
from stc.domain.errors import InvalidInputError

__all__ = ["BenchmarkModel"]

_BATCH_SIZE = 32


class BenchmarkModel:
    """Measure single-utterance latency and batch throughput per thread count."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock

    def run(
        self,
        classifier_factory: Callable[[int], Classifier],
        texts: Sequence[str],
        *,
        threads: Sequence[int],
        runs: int,
        environment: Mapping[str, object],
    ) -> dict[str, object]:
        """Benchmark each thread count and return the S12 JSON payload."""
        if not texts:
            raise InvalidInputError("benchmark needs at least one text")
        if runs < 1:
            raise InvalidInputError("runs must be positive")
        if not threads:
            raise InvalidInputError("threads must contain at least one value")
        results: list[dict[str, object]] = []
        for thread_count in threads:
            classifier = classifier_factory(int(thread_count))
            classifier.predict_scores([texts[0]])  # warm-up
            latencies: list[float] = []
            for index in range(runs):
                text = texts[index % len(texts)]
                started = self._clock()
                classifier.predict_scores([text])
                latencies.append((self._clock() - started) * 1000.0)
            sample = np.asarray(latencies, dtype=np.float64)
            batch = list(texts[:_BATCH_SIZE])
            batch_started = self._clock()
            classifier.predict_scores(batch)
            batch_seconds = max(self._clock() - batch_started, 1e-12)
            results.append(
                {
                    "threads": int(thread_count),
                    "p50_ms": float(np.percentile(sample, 50)),
                    "p95_ms": float(np.percentile(sample, 95)),
                    "mean_ms": float(np.mean(sample)),
                    "batch32_throughput_per_s": len(batch) / batch_seconds,
                    "n_runs": runs,
                }
            )
        return {
            "schema_version": 1,
            "cpu_model": environment.get("cpu_model", "unknown"),
            "onnxruntime_version": environment.get("onnxruntime_version", "unknown"),
            "runs": runs,
            "batch_size": _BATCH_SIZE,
            "threads": results,
        }
