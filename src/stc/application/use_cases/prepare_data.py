"""Fetch, verify, parse, audit, clean and persist the MASSIVE dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from stc.application.dto import DatasetMeta, RawDatasetRef
from stc.application.ports import DatasetParser, DatasetRepository, RawDatasetSource
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.leakage import CleanedTrain, LeakageAudit, audit_splits, clean_train
from stc.domain.splits import Split
from stc.domain.text import normalize_for_dedup


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    """In-memory result of fetching and auditing the raw dataset."""

    ref: RawDatasetRef
    splits: Mapping[Split, tuple[Example, ...]]
    audit: LeakageAudit
    cleaned: CleanedTrain
    stc_version: str


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """What the prepare pipeline persisted and the audit it computed."""

    meta: DatasetMeta
    audit: LeakageAudit


class PrepareDataPipeline:
    """The S1/S2 pipeline: fetch -> verify -> parse -> normalise -> audit -> clean -> persist."""

    def __init__(
        self,
        source: RawDatasetSource,
        parser: DatasetParser,
        repository: DatasetRepository,
        raw_dir: Path,
        *,
        stc_version: str,
        dataset_id: str = "massive",
        version: str = "1.0",
    ) -> None:
        self._source = source
        self._parser = parser
        self._repository = repository
        self._raw_dir = Path(raw_dir)
        self._stc_version = stc_version
        self._dataset_id = dataset_id
        self._version = version

    @property
    def repository(self) -> DatasetRepository:
        """The repository the pipeline persists into."""
        return self._repository

    def inspect(self, *, force: bool = False) -> DatasetInspection:
        """Run every stage except persistence and return the in-memory evidence."""
        ref = self._source.fetch(self._raw_dir, force=force)
        splits = dict(self._parser.parse(ref.root / "es-ES.jsonl"))
        for split, rows in splits.items():
            self._validate_texts(split, rows)
        train = splits[Split.TRAIN]
        dev = splits[Split.DEV]
        test = splits[Split.TEST]
        audit = audit_splits(train, dev, test)
        cleaned = clean_train(train, dev, test)
        return DatasetInspection(
            ref=ref,
            splits=splits,
            audit=audit,
            cleaned=cleaned,
            stc_version=self._stc_version,
        )

    def run(self, *, force: bool = False) -> PrepareResult:
        """Persist the cleaned train split and the untouched dev/test splits."""
        inspection = self.inspect(force=force)
        self._repository.save(Split.TRAIN, inspection.cleaned.rows)
        self._repository.save(Split.DEV, inspection.splits[Split.DEV])
        self._repository.save(Split.TEST, inspection.splits[Split.TEST])

        drops = Counter(row.reason.value for row in inspection.cleaned.dropped)
        meta = DatasetMeta(
            dataset_id=self._dataset_id,
            version=self._version,
            tarball_sha256=inspection.ref.sha256,
            stc_version=self._stc_version,
            raw_counts={split.value: len(rows) for split, rows in inspection.splits.items()},
            processed_counts={
                Split.TRAIN.value: len(inspection.cleaned.rows),
                Split.DEV.value: len(inspection.splits[Split.DEV]),
                Split.TEST.value: len(inspection.splits[Split.TEST]),
            },
            dropped_counts=dict(sorted(drops.items())),
        )
        self._repository.save_meta(meta)
        return PrepareResult(meta=meta, audit=inspection.audit)

    @staticmethod
    def _validate_texts(split: Split, rows: tuple[Example, ...]) -> None:
        for example in rows:
            if not normalize_for_dedup(example.text):
                raise InvalidInputError(
                    f"{split.value} row {example.id}: text is empty after normalisation"
                )
