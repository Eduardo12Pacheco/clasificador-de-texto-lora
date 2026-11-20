"""Merge a trained run and export it to ONNX, plus the D12 variants (S11)."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from stc import __version__
from stc.adapters.export.parity import GateResult, quantization_gate
from stc.adapters.export.quantization import build_variant as _build_variant
from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.domain.errors import ArtifactNotFoundError, IntegrityError, InvalidInputError
from stc.domain.labels import LabelSpace

__all__ = ["OnnxExporter", "quantisation_variants"]

MANIFEST_NAME = "manifest.json"
ONNX_OPSET = 17
BUNDLE_NAME = "stc-es-intent"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ArtifactNotFoundError(f"artifact not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise IntegrityError(f"{path.name} must be a JSON object")
    return cast(Mapping[str, object], payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _base_license(backbone: str) -> str:
    if "MrBERT" in backbone or "Qwen" in backbone or backbone == "stc-tiny-test":
        return "Apache-2.0"
    if "bertin" in backbone:
        return "CC BY 4.0"
    return "unknown"


def quantisation_variants() -> tuple[str, ...]:
    """Expose the D12 candidate names without importing onnx."""
    from stc.adapters.export.quantization import QUANT_VARIANTS

    return QUANT_VARIANTS


class OnnxExporter:
    """Torch/ONNX adapter behind the ``ModelExporter`` port.

    ``base_model_loader`` is injected by the composition root so the exporter
    does not import the training adapter package (and can build the tiny test
    backbone, which is not downloadable from the Hugging Face Hub).
    """

    def __init__(self, base_model_loader: Any | None = None) -> None:
        self._base_model_loader = base_model_loader

    def _load_base(self, backbone: str, n_labels: int, tokenizer: HfTokenizer) -> Any:
        from transformers import AutoModelForSequenceClassification

        if self._base_model_loader is not None:
            return self._base_model_loader(backbone, n_labels, tokenizer)
        return AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=n_labels,
            attn_implementation="eager",
        )

    def _load_model(self, run_dir: Path, metadata: Mapping[str, object]) -> Any:
        import torch  # noqa: F401 - imported for side effects inside the extra
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        backbone = str(metadata["backbone"])
        method = str(metadata["method"])
        n_labels = int(cast(int, metadata["n_intents"]))
        tokenizer = HfTokenizer.from_file(Path(run_dir) / "tokenizer.json")
        if method == "lora":
            base = self._load_base(backbone, n_labels, tokenizer)
            base.config.pad_token_id = tokenizer.pad_id
            adapter = Path(run_dir) / "adapter"
            if not adapter.is_dir():
                raise ArtifactNotFoundError(f"LoRA adapter not found at {adapter}")
            merged = PeftModel.from_pretrained(base, str(adapter))
            model = merged.merge_and_unload()
        elif method == "full":
            model = AutoModelForSequenceClassification.from_pretrained(
                str(Path(run_dir) / "model"),
                attn_implementation="eager",
            )
        else:
            raise InvalidInputError(f"unsupported training method: {method!r}")
        config = getattr(model, "config", None)
        if config is not None:
            config.attn_implementation = "eager"
        model.eval()
        return model

    @staticmethod
    def _scores(model: Any, tokenizer: HfTokenizer, texts: Sequence[str]) -> NDArray[np.float32]:
        import torch

        chunks: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                encoded = tokenizer.encode_batch(texts[start : start + 64])
                ids = torch.as_tensor(encoded.input_ids, dtype=torch.long)
                mask = torch.as_tensor(encoded.attention_mask, dtype=torch.long)
                logits = model(input_ids=ids, attention_mask=mask).logits
                chunks.append(logits.float().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def export(self, run_dir: Path, out_dir: Path) -> Path:
        """Merge ``run_dir``, export fp32 ONNX and write a verifiable bundle."""
        import torch

        source = Path(run_dir)
        target = Path(out_dir)
        metadata = _read_json(source / "run.json")
        labels_payload = _read_json(source / "labels.json")
        raw_intents = labels_payload.get("intents")
        if not isinstance(raw_intents, list) or not all(
            isinstance(item, str) for item in raw_intents
        ):
            raise IntegrityError("run labels.json must contain a list of intent strings")
        labels = LabelSpace.from_intents(cast(list[str], raw_intents))
        tokenizer_file = source / "tokenizer.json"
        tokenizer = HfTokenizer.from_file(tokenizer_file)
        model = self._load_model(source, metadata)

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        model_path = target / "model.onnx"
        encoded = tokenizer.encode_batch(["enciende la luz del cuarto"])
        ids = torch.as_tensor(encoded.input_ids, dtype=torch.long)
        mask = torch.as_tensor(encoded.attention_mask, dtype=torch.long)

        class _LogitsWrapper(torch.nn.Module):  # type: ignore[misc]
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

        wrapper = _LogitsWrapper(model)
        torch.onnx.export(
            wrapper,
            (ids, mask),
            str(model_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        import onnx

        onnx.checker.check_model(str(model_path))
        shutil.copyfile(tokenizer_file, target / "tokenizer.json")
        _write_json(
            target / "labels.json",
            {"schema_version": 1, "intents": list(labels.intents)},
        )

        backbone = str(metadata["backbone"])
        method = str(metadata["method"])
        config = metadata.get("config")
        config_map = config if isinstance(config, Mapping) else {}
        manifest: dict[str, object] = {
            "bundle_version": 1,
            "name": BUNDLE_NAME,
            "version": __version__,
            "kind": "onnx-seqcls",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stc_version": __version__,
            "model": {
                "parameters": int(cast(int, metadata.get("total_params", 0))),
                "trainable_parameters": int(cast(int, metadata.get("trainable_params", 0))),
            },
            "base_model": {
                "id": backbone,
                "revision": "main",
                "license": _base_license(backbone),
            },
            "adapter": {
                "method": method,
                "r": config_map.get("r"),
                "alpha": config_map.get("alpha"),
                "lora_dropout": config_map.get("lora_dropout"),
                "target_modules": "all-linear",
            },
            "dataset": {
                "id": str(metadata.get("dataset", "massive")),
                "counts": {"train": metadata.get("n_train"), "dev": metadata.get("n_dev")},
            },
            "training_run_id": source.name,
            "quantization": {"mode": "fp32"},
            "onnx": {
                "file": "model.onnx",
                "opset": ONNX_OPSET,
                "inputs": ["input_ids", "attention_mask"],
                "outputs": ["logits"],
            },
            "tokenizer": {"max_length": tokenizer.max_length, "pad_id": tokenizer.pad_id},
            "files": [],
        }
        manifest["files"] = self._hash_entries(target)
        _write_json(target / MANIFEST_NAME, manifest)
        return target

    def score_logits(self, run_dir: Path, texts: Sequence[str]) -> NDArray[np.float32]:
        """Score ``texts`` with the merged torch model (parity reference)."""
        source = Path(run_dir)
        metadata = _read_json(source / "run.json")
        tokenizer = HfTokenizer.from_file(source / "tokenizer.json")
        model = self._load_model(source, metadata)
        return self._scores(model, tokenizer, list(texts))

    def build_variant(
        self,
        bundle_dir: Path,
        variant: str,
        calibration_texts: Sequence[str],
        out_path: Path,
    ) -> Path:
        """Quantise the bundle's fp32 ONNX file into ``out_path``."""
        root = Path(bundle_dir)
        manifest = _read_json(root / MANIFEST_NAME)
        onnx_section = manifest.get("onnx")
        relative = onnx_section.get("file") if isinstance(onnx_section, Mapping) else None
        source = root / (relative if isinstance(relative, str) else "model.onnx")
        tokenizer = HfTokenizer.from_file(root / "tokenizer.json")
        return _build_variant(source, variant, calibration_texts, tokenizer, Path(out_path))

    def gate(
        self,
        fp32_scores: NDArray[np.float32],
        variant_scores: NDArray[np.float32],
        y_true: Sequence[int],
    ) -> Mapping[str, object]:
        """Apply the binding macro-F1/agreement gate."""
        result: GateResult = quantization_gate(fp32_scores, variant_scores, y_true)
        return result.to_dict()

    def ship_variant(
        self,
        bundle_dir: Path,
        variant: str,
        variant_path: Path,
    ) -> Path:
        """Move a passing variant into the bundle and repoint the manifest."""
        root = Path(bundle_dir)
        manifest = dict(_read_json(root / MANIFEST_NAME))
        target = root / f"model.{variant}.onnx"
        shutil.copyfile(variant_path, target)
        fp32 = root / "model.onnx"
        fp32.unlink(missing_ok=True)
        onnx_section = dict(cast(Mapping[str, object], manifest.get("onnx", {})))
        onnx_section["file"] = target.name
        manifest["onnx"] = onnx_section
        manifest["quantization"] = {"mode": variant}
        manifest["files"] = self._hash_entries(root)
        _write_json(root / MANIFEST_NAME, manifest)
        return target

    @staticmethod
    def _hash_entries(root: Path) -> list[dict[str, object]]:
        return [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != MANIFEST_NAME
        ]
