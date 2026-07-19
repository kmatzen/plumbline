"""Subprocess-level CLI tests.

These run the installed ``plumbline`` console script as a separate process,
validating the packaging entry point (``[project.scripts]`` in pyproject.toml)
plus any sys.path or import-side-effect surprises that an in-process
``typer.CliRunner`` would miss.

Fast — each subprocess is ~0.5s on a warm uv cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Each test spawns a fresh interpreter (~0.5-5s); marked so a pre-push run can
# deselect them with `-m "not slow"`.
pytestmark = pytest.mark.slow


def _console_script() -> list[str]:
    """Command that runs *this* checkout's CLI.

    Resolve the console script next to the running interpreter rather than
    via ``shutil.which``: a bare PATH lookup finds whatever ``plumbline`` is
    installed globally (pipx, another venv) and would silently test that
    build instead of this one. Falls back to ``python -m plumbline`` — the
    package ships ``__main__.py`` — so running outside ``uv run`` exercises
    the CLI rather than skipping the whole module.
    """
    candidate = Path(sys.executable).parent / ("plumbline.exe" if os.name == "nt" else "plumbline")
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-m", "plumbline"]


PLUMBLINE_CMD = _console_script()


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*PLUMBLINE_CMD, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **(env or {})},
        check=False,
    )


def test_version() -> None:
    r = _run("--version")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("plumbline ")


def test_help() -> None:
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    # Expect every subcommand to appear.
    for cmd in ("list-models", "list-datasets", "run", "reproduce", "clear-cache", "cache-info"):
        assert cmd in r.stdout, f"missing subcommand {cmd!r} in help"


def test_no_args_prints_help() -> None:
    # Should exit 0 and show help (our callback explicitly echoes it).
    r = _run()
    assert r.returncode == 0, r.stderr
    assert "Usage" in r.stdout
    assert "list-models" in r.stdout


def test_list_models_includes_all_v01_adapters() -> None:
    r = _run("list-models")
    assert r.returncode == 0, r.stderr
    # All v0.1 adapters should be present even without their optional deps.
    for name in ("depth-anything-v2", "metric3d-v2", "mast3r", "vggt", "depth-anything-3"):
        assert name in r.stdout, f"missing adapter {name!r}"


def test_list_datasets_includes_all_v01() -> None:
    r = _run("list-datasets")
    assert r.returncode == 0, r.stderr
    for name in ("sintel", "nyuv2", "eth3d"):
        assert name in r.stdout, f"missing dataset {name!r}"


def test_cache_info_empty(tmp_path: Path) -> None:
    r = _run("cache-info", "--cache-dir", str(tmp_path), env={"PLUMBLINE_CACHE_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert "no cache" in r.stdout or "0 entries" in r.stdout


def test_clear_cache_empty(tmp_path: Path) -> None:
    r = _run("clear-cache", "--cache-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "removed 0" in r.stdout or "removed 0 cached" in r.stdout


def test_report_roundtrip(tmp_path: Path) -> None:
    """Write a minimal JSON report and read it back through `plumbline report`."""
    from plumbline.report import Report, RunEnvironment, SampleResult

    report = Report(
        model="fake",
        model_version="1",
        dataset="fake",
        split="test",
        tasks=["mono_depth"],
        scale_alignment="median",
        aggregate_metrics={"abs_rel": 0.123},
        per_sample=[SampleResult("s0", metrics={"abs_rel": 0.123})],
        n_total=1,
        n_evaluated=1,
        environment=RunEnvironment(plumbline_version="0.1.0.dev0"),
    )
    report_path = tmp_path / "r.json"
    report.save_json(report_path)

    # JSON format round-trip.
    r = _run("report", "--json", str(report_path), "--format", "json")
    assert r.returncode == 0, r.stderr
    parsed = json.loads(r.stdout)
    assert parsed["model"] == "fake"
    assert parsed["aggregate_metrics"]["abs_rel"] == 0.123

    # Markdown format round-trip.
    r = _run("report", "--json", str(report_path), "--format", "markdown")
    assert r.returncode == 0, r.stderr
    assert "fake" in r.stdout
    assert "abs_rel" in r.stdout


def test_run_unknown_model_clear_error() -> None:
    r = _run(
        "run",
        "--model",
        "does-not-exist",
        "--dataset",
        "sintel",
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "Unknown model" in combined or "does-not-exist" in combined
