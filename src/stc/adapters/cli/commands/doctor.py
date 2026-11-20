"""``stc doctor``: environment diagnostics, including the optional GPU stack."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from stc import __version__, bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.adapters.cli.rendering import console
from stc.config import Settings


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _uv_lock_sha256() -> str:
    path = Path("uv.lock")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0.0
    return round(float(pages * page_size) / (1024**3), 2)


def _driver_version() -> str:
    try:
        result = subprocess.run(
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _onnxruntime_environment() -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - exercised through monkeypatching
        raise ImportError("onnxruntime is not importable") from error
    return {
        "version": str(ort.__version__),
        "providers": sorted(ort.get_available_providers()),
    }


def _settings_payload(settings: Settings) -> dict[str, object]:
    names = (
        "model",
        "log_level",
        "tracker",
        "mlflow_uri",
        "threads",
        "offline",
        "seed",
        "api_host",
        "api_port",
        "api_max_batch",
        "api_max_chars",
    )
    return {name: getattr(settings, name) for name in names}


def _installed_models_payload(settings: Settings) -> list[dict[str, object]]:
    try:
        installed = bootstrap.build_model_store(settings).list_installed()
    except Exception:
        return []
    return [
        {
            "name": info.model_id,
            "version": info.version,
            "kind": info.kind,
            "quantization": info.quantization,
        }
        for info in installed
    ]


def build_environment_report(*, gpu: bool) -> dict[str, object]:
    """Collect the environment payload documented for T7/T11/T9."""
    settings = bootstrap.build_settings()
    payload: dict[str, object] = {
        "schema_version": 1,
        "stc_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "ram_gib": _ram_gib(),
        "git_sha": _git_sha(),
        "uv_lock_sha256": _uv_lock_sha256(),
        "library_versions": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "optuna",
                "mlflow",
                "onnxruntime",
                "numpy",
            )
        },
        "onnxruntime": _onnxruntime_environment(),
        "settings": _settings_payload(settings),
        "paths": {
            "home": str(settings.resolve_home()),
            "data_dir": str(settings.resolve_data_dir()),
            "models": str(settings.resolve_home() / "models"),
        },
        "installed_models": _installed_models_payload(settings),
    }
    if not gpu:
        payload["gpu_requested"] = False
        return payload
    bootstrap.require_train_extra()
    torch = importlib.import_module("torch")

    available = torch.cuda.is_available()
    payload.update(
        {
            "gpu_requested": True,
            "cuda_available": available,
            "device_name": torch.cuda.get_device_name(0) if available else "cpu",
            "total_memory_mib": (
                round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
                if available
                else 0.0
            ),
            "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
            "driver_version": _driver_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else "none",
        }
    )
    return payload


@handle_domain_errors
def doctor(
    ctx: typer.Context,
    gpu: Annotated[
        bool,
        typer.Option("--gpu", help="Probe CUDA, bf16, driver and training libraries."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the environment report as JSON."),
    ] = False,
) -> None:
    """Print environment diagnostics; ``--gpu`` adds the training stack."""
    try:
        payload = build_environment_report(gpu=gpu)
    except ImportError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if as_json:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    table = Table(title="stc environment")
    table.add_column("key")
    table.add_column("value")
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, dict):
            rendered = ", ".join(f"{name}={version}" for name, version in sorted(value.items()))
        else:
            rendered = str(value)
        table.add_row(key, rendered)
    console(ctx).print(table)


def python_version() -> str:
    """Return the interpreter version (kept for the T9 doctor completion)."""
    return sys.version.split()[0]
