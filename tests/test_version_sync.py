"""Guard against the package version drifting from the build metadata.

`pyproject.toml` (`[project].version`) is the version PyPI/`pip` see; the package
exposes `plumbline.__version__` from `_version.py`. If these drift, an installed
`plumbline-bench 0.2.0` would report `__version__ == "0.1.0"` (an embarrassing
release bug we hit once). Keep them in lockstep.
"""

from __future__ import annotations

import re
from pathlib import Path

from plumbline import __version__

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Read `[project].version` without assuming tomllib (py3.10 lacks it)."""
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]
    text = _PYPROJECT.read_text()
    if tomllib is not None:
        return str(tomllib.loads(text)["project"]["version"])
    # Fallback: first `version = "..."` inside the [project] table.
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped != "[project]":
            in_project = False
        if stripped == "[project]":
            in_project = True
            continue
        if in_project:
            m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if m:
                return m.group(1)
    raise AssertionError("could not find [project].version in pyproject.toml")


def test_package_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version(), (
        f"plumbline.__version__ ({__version__}) != pyproject [project].version "
        f"({_pyproject_version()}). Bump src/plumbline/_version.py to match."
    )
