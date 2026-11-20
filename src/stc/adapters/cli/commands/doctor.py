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
from rich.console import Console
from rich.table import Table

from stc import __version__, bootstrap
from stc.adapters.cli.errors import handle_domain_errors


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


def build_environment_report(*, gpu: bool) -> dict[str, object]:
    """Collect the environment payload documented for T7/T11."""
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
    payload = build_environment_report(gpu=gpu)
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
    Console().print(table)


def python_version() -> str:
    """Return the interpreter version (kept for the T9 doctor completion)."""
    return sys.version.split()[0]
