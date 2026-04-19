"""Smoke tests for the typer CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from plumbline.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "plumbline" in result.stdout


def test_help_no_args() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)  # typer may exit 2 when showing help
    # Help must surface the subcommands.
    assert "list-models" in result.stdout
    assert "run" in result.stdout


def test_list_models_runs() -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    # Rich may wrap the title across lines; look for the words independently.
    assert "Registered" in result.stdout and "models" in result.stdout


def test_list_datasets_runs() -> None:
    result = runner.invoke(app, ["list-datasets"])
    assert result.exit_code == 0
    assert "Registered" in result.stdout and "datasets" in result.stdout


def test_run_unknown_model_errors() -> None:
    # mix_stderr=False is default in newer click; combine explicitly.
    result = runner.invoke(app, ["run", "--model", "no-such", "--dataset", "no-such"])
    assert result.exit_code != 0
    combined = result.output + (result.stderr if hasattr(result, "stderr") else "")
    assert (
        "Unknown" in combined
        or "Invalid value" in combined
        or isinstance(result.exception, SystemExit)
    )


def test_cache_info_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["cache-info", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no cache" in result.stdout


def test_clear_cache_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["clear-cache", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_parse_kv_value_types() -> None:
    from plumbline.cli import _parse_kv_value

    assert _parse_kv_value("42") == 42
    assert _parse_kv_value("3.14") == 3.14
    assert _parse_kv_value("true") is True
    assert _parse_kv_value("False") is False
    assert _parse_kv_value("none") is None
    assert _parse_kv_value("null") is None
    assert _parse_kv_value("hello") == "hello"
    assert _parse_kv_value('"quoted"') == "quoted"
    assert _parse_kv_value("'quoted'") == "quoted"
    # Strings that look like versions stay as strings.
    assert _parse_kv_value("1.2.3") == "1.2.3"
