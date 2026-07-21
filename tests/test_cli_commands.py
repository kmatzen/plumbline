"""Coverage for CLI commands not exercised by test_cli.py / test_cli_subprocess.py.

Targets the previously-untested branches of ``plumbline.cli``: the ``run`` and
``make-samples`` happy paths (via the ``registered_fakes`` fixture), ``queue``'s
table-rendering branches (both plain listing and ``--run``), ``install``'s
plan-printing / ``--all`` / ``--yes`` branches (subprocess mocked out), and
``doctor`` / ``report`` / ``cache-info``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from plumbline import queue as queue_mod
from plumbline.cli import app
from plumbline.report import Report

runner = CliRunner()


def _write(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# run / make-samples
# ---------------------------------------------------------------------------


def test_run_cmd_end_to_end_writes_report(registered_fakes: None, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "test-fixed-depth",
            "--dataset",
            "test-synthetic",
            "--scale-alignment",
            "none",
            "--device",
            "cpu",
            "--subset",
            "2",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["model"] == "test-fixed-depth"
    assert "wrote" in result.output


def test_run_cmd_accepts_dataset_kwargs(registered_fakes: None) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "test-fixed-depth",
            "--dataset",
            "test-synthetic",
            "--scale-alignment",
            "none",
            "--dataset-kwargs",
            "n_samples=2",
        ],
    )
    assert result.exit_code == 0, result.output


def test_run_cmd_bad_dataset_kwargs_missing_equals(registered_fakes: None) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "test-fixed-depth",
            "--dataset",
            "test-synthetic",
            "--dataset-kwargs",
            "not-a-kv-pair",
        ],
    )
    assert result.exit_code != 0


def test_make_samples_writes_file(registered_fakes: None, tmp_path: Path) -> None:
    out = tmp_path / "samples.txt"
    result = runner.invoke(
        app,
        [
            "make-samples",
            "--dataset",
            "test-synthetic",
            "--output",
            str(out),
            "--subset",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "# dataset: test-synthetic" in text
    assert "# subset: 2" in text
    # subset(2) stride-samples the 3-element fake dataset (s0, s1, s2) down to 2.
    assert "s0" in text and "s2" in text


# ---------------------------------------------------------------------------
# report / cache-info
# ---------------------------------------------------------------------------


def test_report_cmd_bad_format(tmp_path: Path) -> None:
    report = Report(
        model="m",
        model_version="1",
        dataset="d",
        split="test",
        tasks=["mono_depth"],
        scale_alignment="none",
        aggregate_metrics={"abs_rel": 0.1},
    )
    path = tmp_path / "r.json"
    report.save_json(path)
    result = runner.invoke(app, ["report", "--json", str(path), "--format", "xml"])
    assert result.exit_code != 0


def test_report_cmd_json_format(tmp_path: Path) -> None:
    report = Report(
        model="m",
        model_version="1",
        dataset="d",
        split="test",
        tasks=["mono_depth"],
        scale_alignment="none",
        aggregate_metrics={"abs_rel": 0.1},
    )
    path = tmp_path / "r.json"
    report.save_json(path)
    result = runner.invoke(app, ["report", "--json", str(path), "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["model"] == "m"


def test_cache_info_nonempty(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "some-model" / "some-dataset"
    pred_dir.mkdir(parents=True)
    (pred_dir / "a.npz").write_bytes(b"0" * 1024)
    (pred_dir / "b.npz").write_bytes(b"0" * 2048)
    result = runner.invoke(app, ["cache-info", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "2 entries" in result.output


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------


@pytest.fixture
def _tmp_queue(repro_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    queue_path = repro_dir / "gpu_queue.yaml"
    monkeypatch.setattr(queue_mod, "QUEUE_PATH", queue_path)
    yield repro_dir


def _write_repro(repro_dir: Path, name: str, *, target: float, value: float) -> None:
    cache_dir = repro_dir / f"cache_{name}"
    _write(
        repro_dir / f"{name}.yaml",
        f"""
name: {name}
model: {{name: test-fixed-depth, kwargs: {{target_abs_rel: {target}}}}}
dataset: {{name: test-synthetic, kwargs: {{n_samples: 2}}}}
tasks: [mono_depth]
scale_alignment: none
device: cpu
cache_dir: {cache_dir}
paper_reference:
  primary_metric: abs_rel
  value: {value}
  tolerance_relative: 0.05
""",
    )


def test_queue_cmd_listing(_tmp_queue: Path, registered_fakes: None) -> None:
    _write_repro(_tmp_queue, "a", target=0.1, value=0.1)
    _write_repro(_tmp_queue, "b", target=0.1, value=0.1)
    _write(
        _tmp_queue / "gpu_queue.yaml",
        """
jobs:
  - {name: a, status: pending, priority: 1, paper_target: "Table 1"}
  - {name: b, status: blocked, priority: 2, blocked_on: "waiting"}
""",
    )
    result = runner.invoke(app, ["queue"])
    assert result.exit_code == 0, result.output
    assert "a" in result.output
    assert "pending" in result.output
    # blocked job hidden by default
    assert "waiting" not in result.output
    assert "1 pending, 1 blocked, 0 done" in result.output


def test_queue_cmd_listing_include_blocked(_tmp_queue: Path, registered_fakes: None) -> None:
    _write_repro(_tmp_queue, "a", target=0.1, value=0.1)
    _write_repro(_tmp_queue, "b", target=0.1, value=0.1)
    _write(
        _tmp_queue / "gpu_queue.yaml",
        """
jobs:
  - {name: a, status: pending, priority: 1}
  - {name: b, status: blocked, priority: 2, blocked_on: "waiting"}
""",
    )
    result = runner.invoke(app, ["queue", "--include-blocked"])
    assert result.exit_code == 0, result.output
    assert "b" in result.output


def test_queue_cmd_run_no_pending(_tmp_queue: Path, registered_fakes: None) -> None:
    _write(_tmp_queue / "gpu_queue.yaml", "jobs: []\n")
    result = runner.invoke(app, ["queue", "--run"])
    assert result.exit_code == 0, result.output
    assert "no pending jobs" in result.output


def test_queue_cmd_run_writes_results(
    _tmp_queue: Path, registered_fakes: None, tmp_path: Path
) -> None:
    _write_repro(_tmp_queue, "good", target=0.1, value=0.1)
    _write_repro(_tmp_queue, "bad", target=0.2, value=0.1)
    _write(
        _tmp_queue / "gpu_queue.yaml",
        """
jobs:
  - {name: good, status: pending, priority: 1}
  - {name: bad, status: pending, priority: 2}
""",
    )
    out = tmp_path / "summary.json"
    result = runner.invoke(app, ["queue", "--run", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "match" in result.output
    assert "mismatch" in result.output
    assert "1/2 job(s) matched" in result.output
    assert out.exists()


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def test_install_cmd_list_no_args() -> None:
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 0
    assert "Adapter install plans" in result.output


def test_install_cmd_unknown_name() -> None:
    result = runner.invoke(app, ["install", "not-a-real-adapter"])
    assert result.exit_code != 0


def test_install_cmd_single_name_plan_only() -> None:
    result = runner.invoke(app, ["install", "vggt"])
    assert result.exit_code == 0
    assert "install plan for vggt" in result.output
    assert "--yes" in result.output


def test_install_cmd_single_name_yes_runs_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    monkeypatch.setattr("plumbline.cli.subprocess.run", mock_run)
    result = runner.invoke(app, ["install", "vggt", "--yes"])
    assert result.exit_code == 0, result.output
    assert mock_run.called
    assert "installed" in result.output


def test_install_cmd_yes_step_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=1)
    monkeypatch.setattr("plumbline.cli.subprocess.run", mock_run)
    result = runner.invoke(app, ["install", "vggt", "--yes"])
    assert result.exit_code != 0


def test_install_cmd_all_plan_only() -> None:
    result = runner.invoke(app, ["install", "--all"])
    assert result.exit_code == 0
    assert "Re-run with" in result.output


def test_install_cmd_all_yes_runs_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    monkeypatch.setattr("plumbline.cli.subprocess.run", mock_run)
    result = runner.invoke(app, ["install", "--all", "--yes"])
    assert result.exit_code == 0, result.output
    assert mock_run.called


def test_install_cmd_no_steps_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any "base"-kind adapter has an empty install plan.
    from plumbline.install import INSTALL_SPECS

    base_names = [n for n, s in INSTALL_SPECS.items() if s.kind == "base"]
    if not base_names:
        pytest.skip("no base-kind adapter registered")
    result = runner.invoke(app, ["install", base_names[0]])
    assert result.exit_code == 0
    assert "nothing to install" in result.output


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_cmd_all() -> None:
    result = runner.invoke(app, ["doctor"])
    assert "Adapter doctor" in result.output
    # Exit code depends on whether every adapter is installed in this env;
    # both success and "some missing" (1) are valid outcomes here.
    assert result.exit_code in (0, 1)


def test_doctor_cmd_unknown_name() -> None:
    result = runner.invoke(app, ["doctor", "not-a-real-adapter"])
    assert result.exit_code != 0


def test_doctor_cmd_single_missing_adapter() -> None:
    from plumbline.install import INSTALL_SPECS

    # vggt is a pip/git adapter not installed in the base test env.
    assert "vggt" in INSTALL_SPECS
    result = runner.invoke(app, ["doctor", "vggt"])
    assert result.exit_code in (0, 1)
    assert "vggt" in result.output
