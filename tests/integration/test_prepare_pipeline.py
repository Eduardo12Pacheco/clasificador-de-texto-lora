"""End-to-end prepare/audit pipeline on a fixture-shaped synthetic archive."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.adapters.data.massive_archive import MassiveArchiveSource
from stc.adapters.data.massive_parser import MassiveParser
from stc.application.use_cases.audit_data import AuditDataUseCase
from stc.application.use_cases.prepare_data import PrepareDataPipeline
from stc.domain.splits import Split

pytestmark = pytest.mark.integration

RAW_ROWS = [
    ("1", "train", "play_music", "play", "Pon música", "w1"),
    ("2", "train", "play_music", "play", "pon musica", "w2"),
    ("3", "train", "open_door", "open", "Abre la puerta", "w1"),
    ("4", "train", "close_door", "close", "abre la puerta", "w2"),
    ("5", "train", "alarm_set", "alarm", "texto de dev", "w1"),
    ("6", "train", "alarm_set", "alarm", "texto de test", "w1"),
    ("7", "train", "alarm_set", "alarm", "único train", "w2"),
    ("8", "train", "weather_query", "weather", "otro train", "w2"),
    ("100", "dev", "alarm_set", "alarm", "texto de dev", "w3"),
    ("200", "test", "alarm_set", "alarm", "texto de test", "w3"),
]


def make_tarball(tmp_path: Path) -> tuple[Path, str, int]:
    lines = []
    for row_id, partition, intent, scenario, text, worker in RAW_ROWS:
        lines.append(
            json.dumps(
                {
                    "id": row_id,
                    "locale": "es-ES",
                    "partition": partition,
                    "scenario": scenario,
                    "intent": intent,
                    "utt": text,
                    "annot_utt": f"[{intent}] {text}",
                    "worker_id": worker,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    payload = "".join(lines).encode()
    path = tmp_path / "massive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {
            "1.0/data/es-ES.jsonl": payload,
            "1.0/LICENSE": b"CC BY 4.0\n",
            "1.0/NOTICE.md": b"NOTICE\n",
        }.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    data = path.read_bytes()
    return path, hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture
def pipeline(tmp_path: Path) -> PrepareDataPipeline:
    path, sha256, size = make_tarball(tmp_path)
    source = MassiveArchiveSource(url=path.as_uri(), sha256=sha256, size=size)
    repository = JsonlDatasetRepository(tmp_path / "processed")
    return PrepareDataPipeline(
        source=source,
        parser=MassiveParser(),
        repository=repository,
        raw_dir=tmp_path / "raw" / "massive-1.0",
        stc_version="0.1.0",
    )


def test_prepare_writes_cleaned_splits_and_meta(pipeline: PrepareDataPipeline) -> None:
    result = pipeline.run()

    assert result.audit.train_rows == 8
    assert result.audit.dev_rows == 1
    assert result.audit.test_rows == 1
    assert result.audit.dev_rows_in_train == 1
    assert result.audit.test_rows_in_train == 1
    assert result.audit.train_duplicate_groups == 2
    assert result.meta.raw_counts == {"train": 8, "dev": 1, "test": 1}
    assert result.meta.processed_counts == {"train": 3, "dev": 1, "test": 1}
    assert result.meta.dropped_counts == {
        "EXACT_DUPLICATE": 1,
        "LABEL_CONFLICT_DUPLICATE": 2,
        "OVERLAP_WITH_EVAL": 2,
    }
    assert result.meta.tarball_sha256
    assert result.meta.stc_version == "0.1.0"

    repository = pipeline.repository
    assert [example.id for example in repository.load(Split.TRAIN)] == ["1", "7", "8"]
    assert [example.id for example in repository.load(Split.DEV)] == ["100"]
    assert [example.id for example in repository.load(Split.TEST)] == ["200"]
    assert repository.meta() == result.meta


def test_prepare_is_idempotent(pipeline: PrepareDataPipeline) -> None:
    pipeline.run()
    root = pipeline.repository.root  # type: ignore[attr-defined]
    first = {path.name: path.read_bytes() for path in root.iterdir()}

    second = pipeline.run()

    assert {path.name: path.read_bytes() for path in root.iterdir()} == first
    assert second.meta == pipeline.repository.meta()


def test_inspect_reports_the_dropped_ids_by_reason(pipeline: PrepareDataPipeline) -> None:
    inspection = pipeline.inspect()

    reasons = {row.id: row.reason.value for row in inspection.cleaned.dropped}
    assert reasons == {
        "2": "EXACT_DUPLICATE",
        "3": "LABEL_CONFLICT_DUPLICATE",
        "4": "LABEL_CONFLICT_DUPLICATE",
        "5": "OVERLAP_WITH_EVAL",
        "6": "OVERLAP_WITH_EVAL",
    }


def test_audit_use_case_writes_json_and_markdown(
    pipeline: PrepareDataPipeline, tmp_path: Path
) -> None:
    out_dir = tmp_path / "reports"

    paths = AuditDataUseCase(pipeline=pipeline, out_dir=out_dir).run()

    assert paths.json_path == out_dir / "data_audit.json"
    assert paths.markdown_path == out_dir / "data_audit.md"
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["dev_test_overlap_texts"] == 0
    assert payload["dev_rows_in_train"] == 1
    assert payload["dev_rows_in_train_label_conflict"] == 0
    assert payload["test_rows_in_train"] == 1
    assert payload["test_rows_in_train_label_conflict"] == 0
    assert payload["train_duplicate_groups"] == 2
    assert payload["dev_duplicate_groups"] == 0
    assert payload["test_duplicate_groups"] == 0
    assert payload["drops"] == {
        "EXACT_DUPLICATE": 1,
        "LABEL_CONFLICT_DUPLICATE": 2,
        "OVERLAP_WITH_EVAL": 2,
    }
    assert payload["label_conflict_groups"] == 1
    assert payload["train_after"] == 3
    assert payload["dropped_ids_by_reason"]["OVERLAP_WITH_EVAL"] == ["5", "6"]
    assert payload["worker_partition_overlap"] == {
        "workers_in_multiple_partitions": 1,
        "total_workers": 3,
    }
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "# Data audit" in markdown
    assert "10,898" not in markdown
    assert "| train | 8 | 3 |" in markdown
