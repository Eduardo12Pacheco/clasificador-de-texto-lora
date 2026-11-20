"""Bundle layout, manifest validation and integrity checks (D13).

A bundle is a directory with a ``manifest.json`` that hashes every content file.
The manifest is the trust anchor: loading a bundle never trusts a file whose
sha256 or size does not match the manifest, and unknown extra files are
reported instead of silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

from stc.application.dto import ModelInfo
from stc.domain.calibration import Calibration
from stc.domain.errors import IntegrityError
from stc.domain.labels import LabelSpace

__all__ = [
    "BUNDLE_VERSION",
    "MANIFEST_NAME",
    "OPTIONAL_COMPANIONS",
    "BundleArchiveAdapter",
    "add_calibration",
    "bundle_info",
    "load_calibration",
    "load_labels",
    "model_file",
    "read_manifest",
    "refresh_manifest",
    "safe_member_name",
    "sha256_file",
    "unpack_zip",
    "verify_bundle",
    "write_bundle_zip",
]

MANIFEST_NAME = "manifest.json"
BUNDLE_VERSION = 1
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# Files a bundle may gain after the exporter wrote its manifest. They are
# generated locally by ``stc calibrate`` or added by ``stc bundle pack``; a
# pack always re-hashes everything, so the manifest remains the trust anchor.
OPTIONAL_COMPANIONS = frozenset(
    {
        "calibration.json",
        "baseline/calibration.json",
        "MODEL_CARD.md",
        "LICENSE",
        "NOTICE",
    }
)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise IntegrityError(f"{description} not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{description} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise IntegrityError(f"{description} must be a JSON object")
    return cast(Mapping[str, object], payload)


def read_manifest(bundle: Path) -> Mapping[str, object]:
    """Read and shape-check ``manifest.json`` without hashing the files."""
    root = Path(bundle)
    if not root.is_dir():
        raise IntegrityError(f"bundle directory not found at {root}")
    manifest = _read_json(root / MANIFEST_NAME, "bundle manifest")
    bundle_version = manifest.get("bundle_version")
    if bundle_version != BUNDLE_VERSION:
        raise IntegrityError(
            f"unsupported bundle_version {bundle_version!r}; expected {BUNDLE_VERSION}"
        )
    for field in ("name", "version", "kind"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise IntegrityError(f"manifest {field} must be a non-empty string")
    if manifest["kind"] not in {"onnx-seqcls", "tfidf-linear"}:
        raise IntegrityError(f"unsupported bundle kind: {manifest['kind']!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise IntegrityError("manifest files must be a list")
    return manifest


def _entry_path(entry: Mapping[str, object]) -> str:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise IntegrityError("manifest files entries need a non-empty path")
    _safe_relative(relative)
    return relative


def _safe_relative(name: str) -> PurePosixPath:
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in name or not name:
        raise IntegrityError(f"refusing unsafe bundle path: {name!r}")
    return candidate


def safe_member_name(name: str) -> PurePosixPath:
    """Validate a zip member path and return it as a POSIX path (zip-slip guard)."""
    return _safe_relative(name)


def verify_bundle(bundle: Path) -> Mapping[str, object]:
    """Verify every manifest hash and report missing or unexpected files."""
    root = Path(bundle)
    manifest = read_manifest(root)
    listed: set[str] = set()
    entries = cast(Sequence[object], manifest["files"])
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise IntegrityError("manifest files entries must be objects")
        entry = cast(Mapping[str, object], raw)
        relative = _entry_path(entry)
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise IntegrityError(f"manifest entry {relative} needs a sha256")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise IntegrityError(f"manifest entry {relative} needs a size")
        artifact = root / relative
        if not artifact.is_file():
            raise IntegrityError(f"manifest lists a missing file: {relative}")
        if artifact.stat().st_size != expected_size:
            raise IntegrityError(f"size mismatch for {relative}")
        if sha256_file(artifact) != expected_sha:
            raise IntegrityError(f"checksum mismatch for {relative}")
        listed.add(relative)

    extras = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix() not in listed
        and path.relative_to(root).as_posix() != MANIFEST_NAME
        and path.relative_to(root).as_posix() not in OPTIONAL_COMPANIONS
    ]
    if extras:
        raise IntegrityError("bundle contains unexpected file(s): " + ", ".join(extras))
    return manifest


def load_labels(bundle: Path) -> LabelSpace:
    """Read ``labels.json`` and build the label space."""
    payload = _read_json(Path(bundle) / "labels.json", "bundle labels")
    intents = payload.get("intents")
    if not isinstance(intents, list) or not all(isinstance(item, str) for item in intents):
        raise IntegrityError("labels.json must contain a list of intent strings")
    return LabelSpace.from_intents(cast(list[str], intents))


def load_calibration(bundle: Path) -> Calibration | None:
    """Read ``calibration.json`` when present."""
    path = Path(bundle) / "calibration.json"
    if not path.is_file():
        return None
    return Calibration.from_dict(_read_json(path, "bundle calibration"))


def model_file(bundle: Path, manifest: Mapping[str, object]) -> Path:
    """Resolve the shipped ONNX file declared by the manifest."""
    onnx_section = manifest.get("onnx")
    declared = onnx_section.get("file") if isinstance(onnx_section, Mapping) else None
    relative = declared if isinstance(declared, str) and declared else "model.onnx"
    _safe_relative(relative)
    path = Path(bundle) / relative
    if not path.is_file():
        raise IntegrityError(f"ONNX model not found in bundle: {relative}")
    return path


def bundle_info(bundle: Path, manifest: Mapping[str, object], labels: LabelSpace) -> ModelInfo:
    """Build the cheap :class:`ModelInfo` description of a bundle."""
    model_section = manifest.get("model")
    parameters = model_section.get("parameters") if isinstance(model_section, Mapping) else None
    quantization = manifest.get("quantization")
    mode = quantization.get("mode") if isinstance(quantization, Mapping) else None
    base_model = manifest.get("base_model")
    license_name = base_model.get("license") if isinstance(base_model, Mapping) else None
    return ModelInfo(
        model_id=str(manifest["name"]),
        kind=str(manifest["kind"]),
        version=str(manifest["version"]),
        quantization=str(mode) if isinstance(mode, str) else None,
        parameters=(
            parameters if isinstance(parameters, int) and not isinstance(parameters, bool) else None
        ),
        license=str(license_name) if isinstance(license_name, str) else "unknown",
        n_intents=labels.n_intents,
        n_scenarios=labels.n_scenarios,
    )


def _hash_entries(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    root_manifest = root / MANIFEST_NAME
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == root_manifest:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return entries


def refresh_manifest(bundle: Path) -> Mapping[str, object]:
    """Re-hash every content file of a bundle and rewrite its manifest.

    Used after ``stc calibrate`` adds ``calibration.json`` and by the packer
    after it assembles the release layout. Metadata is preserved.
    """
    root = Path(bundle)
    manifest = dict(read_manifest(root))
    manifest["files"] = _hash_entries(root)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def add_calibration(bundle: Path, calibration: Calibration) -> Path:
    """Write ``calibration.json`` into ``bundle`` and re-hash the manifest."""
    root = Path(bundle)
    payload: dict[str, object] = {
        "schema_version": 1,
        **calibration.to_dict(),
    }
    path = root / "calibration.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    refresh_manifest(root)
    return path


def unpack_zip(archive: Path, dest: Path) -> Path:
    """Safely extract a bundle zip into ``dest`` and return the bundle root.

    Rejects absolute paths, ``..`` traversal, backslashes and symlinks before a
    single byte is extracted, so a hostile archive cannot write outside ``dest``.
    """
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            for name in names:
                safe_member_name(name)
            for info in handle.infolist():
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise IntegrityError(f"refusing symlink in bundle archive: {info.filename}")
            handle.extractall(target)
    except zipfile.BadZipFile as error:
        raise IntegrityError(f"{archive} is not a readable zip archive") from error
    roots = [path for path in target.iterdir() if path.is_dir()]
    if (target / MANIFEST_NAME).is_file():
        return target
    if len(roots) == 1 and (roots[0] / MANIFEST_NAME).is_file():
        return roots[0]
    raise IntegrityError("bundle archive does not contain a manifest.json at its root")


def copy_tree(source: Path, dest: Path) -> None:
    """Copy a bundle directory into ``dest``, replacing an existing tree."""
    target = Path(dest)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        staging = Path(scratch) / target.name
        shutil.copytree(source, staging)
        staging.replace(target)


def write_bundle_zip(bundle: Path, out: Path) -> Path:
    """Write a deterministic deflate zip of ``bundle`` (sorted, fixed mtime)."""
    root = Path(bundle)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())
    return target


class BundleArchiveAdapter:
    """Filesystem implementation of the application ``BundleArchive`` port."""

    def verify(self, bundle_dir: Path) -> Mapping[str, object]:
        """Verify every manifest hash and return the manifest."""
        return verify_bundle(bundle_dir)

    def unpack(self, archive: Path, dest: Path) -> Path:
        """Safely extract a bundle archive and return its root."""
        return unpack_zip(archive, dest)

    def write_zip(self, bundle_dir: Path, out: Path) -> Path:
        """Write a deterministic bundle zip."""
        return write_bundle_zip(bundle_dir, out)

    def refresh_manifest(self, bundle_dir: Path) -> Mapping[str, object]:
        """Re-hash every content file of a bundle."""
        return refresh_manifest(bundle_dir)

    def read_manifest(self, bundle_dir: Path) -> Mapping[str, object]:
        """Read and shape-check a manifest without hashing the files."""
        return read_manifest(bundle_dir)

    def load_labels(self, bundle_dir: Path) -> LabelSpace:
        """Read a bundle label space."""
        return load_labels(bundle_dir)

    def bundle_info(
        self, bundle_dir: Path, manifest: Mapping[str, object], labels: LabelSpace
    ) -> ModelInfo:
        """Describe a bundle without loading its weights."""
        return bundle_info(bundle_dir, manifest, labels)

    def copy_tree(self, source: Path, dest: Path) -> None:
        """Copy a verified bundle directory."""
        copy_tree(source, dest)
