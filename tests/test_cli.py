"""Smoke tests for the typer CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from plumbline.cli import _unknown, app

runner = CliRunner()


def test_unknown_suggests_close_match() -> None:
    # A single-character slip should get a direct "Did you mean" nudge.
    err = _unknown("dataset", "nyuv", ["nyuv2", "kitti", "bonn"])
    assert "Did you mean 'nyuv2'?" in str(err)
    # The full known list still follows the hint.
    assert "Known:" in str(err) and "kitti" in str(err)


def test_unknown_no_suggestion_for_gibberish() -> None:
    # Nothing close enough → no misleading suggestion, just the known list.
    err = _unknown("model", "zzzzzz", ["vggt", "moge", "dust3r"])
    assert "Did you mean" not in str(err)
    assert "Known:" in str(err)


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


def test_install_base_complete_vendored_adapter_is_noop() -> None:
    result = runner.invoke(app, ["install", "depth-anything-3", "--yes"])
    assert result.exit_code == 0
    assert "nothing to install" in " ".join(result.stdout.split())
    assert "export DA3_ROOT" not in result.stdout


def test_install_manual_weights_are_actionable_not_fake_steps() -> None:
    result = runner.invoke(app, ["install", "cut3r"])
    assert result.exit_code == 0
    assert "weights:" in result.stdout
    assert "CUT3R_CKPT" in result.stdout
    assert "export CUT3R_CKPT" not in result.stdout


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


def test_run_typo_model_suggests() -> None:
    result = runner.invoke(app, ["run", "--model", "vgt", "--dataset", "nyuv2"])
    assert result.exit_code != 0
    combined = result.output + (result.stderr if hasattr(result, "stderr") else "")
    # Rich wraps the error box; normalise whitespace before matching.
    normalised = " ".join(combined.split())
    assert "Did you mean 'vggt'?" in normalised


def test_cache_info_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["cache-info", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no cache" in result.stdout


def test_clear_cache_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["clear-cache", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_tracking_review_render_writes_gated_artifact(tmp_path: Path) -> None:
    n = 8
    reference = np.repeat(np.eye(4)[None], n, axis=0)
    reference[:, 0, 3] = np.linspace(0.0, 1.0, n)
    predicted = reference.copy()
    predicted[:, 1, 3] = np.linspace(0.0, 0.1, n)
    pred_path = tmp_path / "pred.npy"
    ref_path = tmp_path / "ref.npy"
    np.save(pred_path, predicted)
    np.save(ref_path, reference)

    result = runner.invoke(
        app,
        [
            "tracking-review",
            "render",
            "--predicted",
            str(pred_path),
            "--reference",
            str(ref_path),
            "--output-root",
            str(tmp_path / "review"),
            "--dataset",
            "tum-dynamics",
            "--sequence",
            "walking_xyz",
            "--system",
            "method-a",
            "--config-hash",
            "abc",
        ],
    )
    assert result.exit_code == 0, result.output
    manifests = list((tmp_path / "review").rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["gate"]["ok"] is True
    assert (manifests[0].parent / "diagnostic.png").is_file()


def test_tracking_review_rank_reads_jsonl_and_reports_both_bad(tmp_path: Path) -> None:
    comparisons = tmp_path / "comparisons.jsonl"
    rows = [
        {"task_id": "s1", "system_a": "strong", "system_b": "weak", "winner": "a"},
        {"task_id": "s2", "system_a": "strong", "system_b": "weak", "winner": "a"},
        {
            "task_id": "s3",
            "system_a": "strong",
            "system_b": "weak",
            "winner": "both_bad",
        },
    ]
    comparisons.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = runner.invoke(
        app,
        [
            "tracking-review",
            "rank",
            "--comparisons",
            str(comparisons),
            "--bootstrap",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["n_comparisons"] == 3
    assert payload["n_rankable"] == 2
    assert payload["ratings"]["strong"]["score"] > payload["ratings"]["weak"]["score"]


def test_tracking_review_pair_and_vote_commands(tmp_path: Path) -> None:
    from plumbline.tracking_review import write_tracking_review_artifact

    n = 8
    reference = np.repeat(np.eye(4)[None], n, axis=0)
    reference[:, 0, 3] = np.linspace(0.0, 1.0, n)
    first = write_tracking_review_artifact(
        reference,
        reference,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="first",
        config_hash="a",
    )
    second_poses = reference.copy()
    second_poses[:, 1, 3] = np.linspace(0.0, 0.1, n)
    second = write_tracking_review_artifact(
        second_poses,
        reference,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="second",
        config_hash="b",
    )
    pair_dir = tmp_path / "pair"
    result = runner.invoke(
        app,
        [
            "tracking-review",
            "pair",
            "--artifact-a",
            str(first),
            "--artifact-b",
            str(second),
            "--output",
            str(pair_dir),
            "--task-id",
            "tum/walk",
            "--comparison-id",
            "pair-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (pair_dir / "review.png").is_file()

    comparisons = tmp_path / "comparisons.jsonl"
    result = runner.invoke(
        app,
        [
            "tracking-review",
            "vote",
            "--pair",
            str(pair_dir / "pair.json"),
            "--winner",
            "tie",
            "--comparisons",
            str(comparisons),
            "--reviewer",
            "tester",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(comparisons.read_text())["winner"] == "tie"


def test_tracking_review_calibrate_command(tmp_path: Path) -> None:
    human = tmp_path / "human.jsonl"
    judge = tmp_path / "judge.jsonl"
    human.write_text(json.dumps({"comparison_id": "1", "winner": "a", "source": "human"}) + "\n")
    judge.write_text(json.dumps({"comparison_id": "1", "winner": "a", "source": "vlm"}) + "\n")
    result = runner.invoke(
        app,
        [
            "tracking-review",
            "calibrate",
            "--human",
            str(human),
            "--judge",
            str(judge),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["coverage"] == 1.0
    assert payload["accuracy"] == 1.0


def test_python_m_entrypoint() -> None:
    # ``python -m plumbline`` must work even where the console script isn't on
    # PATH (relies only on __main__.py + the package being importable).
    r = subprocess.run(
        [sys.executable, "-m", "plumbline", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("plumbline ")


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
