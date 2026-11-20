"""Enforce the hexagonal dependency rules from the architecture."""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "src" / "stc" / "domain"
CONTRACTS_LINE = re.compile(r"Contracts: (\d+) kept, (\d+) broken")


def test_import_linter_contracts_are_kept() -> None:
    executable = shutil.which("lint-imports")
    assert executable is not None, "lint-imports is missing; run `uv sync --group dev`"

    result = subprocess.run(
        [executable, "--no-cache"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, f"lint-imports failed:\n{result.stdout}\n{result.stderr}"
    match = CONTRACTS_LINE.search(result.stdout)
    assert match is not None, result.stdout
    kept, broken = (int(value) for value in match.groups())
    assert broken == 0
    assert kept >= 6


def _domain_sources() -> list[Path]:
    return sorted(DOMAIN_ROOT.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            modules.add(node.module)
    return modules


def _is_allowed(module: str) -> bool:
    top_level = module.split(".", maxsplit=1)[0]
    if top_level in sys.stdlib_module_names:
        return True
    if top_level == "numpy":
        return True
    return module == "stc.domain" or module.startswith("stc.domain.")


def test_domain_imports_only_stdlib_numpy_and_itself() -> None:
    sources = _domain_sources()
    assert sources, "no domain modules found"

    violations = [
        f"{path.relative_to(PROJECT_ROOT)} imports {module}"
        for path in sources
        for module in sorted(_imported_modules(path))
        if not _is_allowed(module)
    ]

    assert not violations, "Domain purity violated:\n" + "\n".join(violations)
