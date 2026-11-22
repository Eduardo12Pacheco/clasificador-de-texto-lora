"""Every ``uv run stc`` command in the training guide must exist in the CLI."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "docs" / "TRAINING.md"
FENCE = re.compile(r"^```(\w*)\s*$")


def _bash_commands(text: str) -> list[str]:
    """Return logical commands from the ```bash fences, joining continuations."""
    commands: list[str] = []
    in_bash = False
    for raw in text.splitlines():
        fence = FENCE.match(raw)
        if fence:
            in_bash = in_bash is False and fence.group(1) == "bash"
            continue
        if not in_bash:
            continue
        line = raw.split(" #", maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if commands and commands[-1].endswith("\\"):
            commands[-1] = commands[-1][:-1].rstrip() + " " + line
        else:
            commands.append(line)
    return commands


def _subcommand(command: str) -> tuple[str, ...] | None:
    try:
        tokens = shlex.split(command)
    except ValueError as error:  # pragma: no cover - defensive on malformed docs
        raise AssertionError(f"cannot parse documented command {command!r}: {error}") from error
    if tokens[:3] != ["uv", "run", "stc"]:
        return None
    remainder = tokens[3:]
    if not remainder or remainder[0] in {"--help", "--version"}:
        return None
    path: list[str] = []
    for token in remainder:
        if token.startswith("-"):
            break
        path.append(token)
    return tuple(path) if path else None


def test_training_guide_commands_exist() -> None:
    commands = _bash_commands(TRAINING.read_text(encoding="utf-8"))
    stc_commands = [(command, _subcommand(command)) for command in commands if _subcommand(command)]

    assert len(stc_commands) >= 8, "expected the documented GPU pipeline commands"
    for command, path in stc_commands:
        assert path is not None
        result = subprocess.run(
            [sys.executable, "-m", "stc", *path, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{command!r} is not a valid CLI path: {result.stderr}"


def test_training_guide_documents_measured_stages() -> None:
    text = TRAINING.read_text(encoding="utf-8")

    for stage in ("screening", "fine-tuning completo", "búsqueda con Optuna", "semillas finales"):
        assert stage.lower() in text.lower()
    for seconds in ("1972.199", "506.509", "1315.813", "754.476"):
        assert seconds in text
