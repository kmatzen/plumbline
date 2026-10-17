from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from PIL import Image

from plumbline.tracking_review import (
    Comparison,
    bootstrap_ratings,
    build_review_pair,
    calibrate_judge_votes,
    choose_balanced_pairs,
    fit_bradley_terry,
    record_review_vote,
    render_trajectory_diagnostic,
    run_external_judge,
    trajectory_diagnostics,
    validate_trajectory_pair,
    write_tracking_review_artifact,
)


def _round_robin() -> list[Comparison]:
    rows: list[Comparison] = []
    for task in ("easy", "medium", "hard"):
        rows.extend(
            [
                Comparison(task, "strong", "middle", "a"),
                Comparison(task, "strong", "weak", "a"),
                Comparison(task, "middle", "weak", "a"),
            ]
        )
    return rows


def test_bradley_terry_recovers_order_and_centers_scores() -> None:
    scores = fit_bradley_terry(_round_robin())
    assert scores["strong"] > scores["middle"] > scores["weak"]
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-9)


def test_ties_produce_equal_ratings() -> None:
    scores = fit_bradley_terry([Comparison("scene", "a", "b", "tie")])
    assert scores["a"] == pytest.approx(scores["b"])


def test_disconnected_comparison_graph_is_rejected() -> None:
    rows = [Comparison("x", "a", "b", "a"), Comparison("y", "c", "d", "a")]
    with pytest.raises(ValueError, match="disconnected"):
        fit_bradley_terry(rows)


def test_bootstrap_is_task_grouped_and_reproducible() -> None:
    first = bootstrap_ratings(_round_robin(), iterations=100, seed=7)
    second = bootstrap_ratings(_round_robin(), iterations=100, seed=7)
    assert first == second
    assert first["strong"].ci_low is not None
    assert first["strong"].ci_high is not None


def test_balanced_pair_selection_prefers_unseen_pairs() -> None:
    existing = [Comparison("old", "a", "b", "a")]
    pairs = choose_balanced_pairs(["c", "b", "a"], existing, count=2)
    assert pairs == [("a", "c"), ("b", "c")]


def test_invalid_winner_is_rejected() -> None:
    with pytest.raises(ValueError, match="winner"):
        Comparison("task", "a", "b", "invalid")


def test_both_bad_is_recorded_but_excluded_from_ranking() -> None:
    rows = [
        Comparison("x", "a", "b", "a"),
        Comparison("y", "a", "b", "both_bad"),
    ]
    assert fit_bradley_terry(rows) == fit_bradley_terry(rows[:1])


def _trajectory(n: int = 20) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    t = np.linspace(0.0, 1.0, n)
    poses[:, 0, 3] = t
    poses[:, 1, 3] = 0.2 * np.sin(2 * np.pi * t)
    poses[:, 2, 3] = t**2
    return poses


def test_trajectory_diagnostics_remove_global_similarity() -> None:
    gt = _trajectory()
    pred = gt.copy()
    pred[:, :3, 3] = 3.0 * pred[:, :3, 3] + np.array([4.0, -2.0, 8.0])
    diagnostics = trajectory_diagnostics(pred, gt)
    np.testing.assert_allclose(diagnostics.translation_error_normalized, 0.0, atol=1e-12)
    np.testing.assert_allclose(diagnostics.rotation_error_deg, 0.0, atol=1e-6)


def test_stationary_reference_does_not_collapse_moving_prediction_to_perfection(tmp_path):
    stationary = np.repeat(np.eye(4)[None], 20, axis=0)
    with pytest.raises(ValueError, match="reference trajectory has no translational extent"):
        trajectory_diagnostics(_trajectory(), stationary)
    artifacts = [
        write_tracking_review_artifact(
            poses, stationary, tmp_path, dataset="d", sequence="s", system=name, config_hash="c"
        )
        for name, poses in (("still", stationary), ("moving", _trajectory()))
    ]
    for artifact in artifacts:
        manifest = json.loads((artifact / "manifest.json").read_text())
        assert manifest["gate"]["ok"]
        assert manifest["files"]["diagnostic"] is None
        assert "no translational extent" in manifest["diagnostic_unavailable_reason"]
        assert json.loads((artifact / "metrics.json").read_text()) == {}
    with pytest.raises(ValueError, match="do not infer a winner"):
        build_review_pair(*artifacts, tmp_path / "pair", task_id="s", comparison_id="stationary")


def test_render_trajectory_diagnostic_is_deterministic(tmp_path) -> None:
    gt = _trajectory()
    pred = gt.copy()
    pred[10:, 0, 3] += 0.1
    first = render_trajectory_diagnostic(pred, gt, tmp_path / "first.png")
    second = render_trajectory_diagnostic(pred, gt, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.size == (1200, 720)


def test_gate_accepts_frozen_but_rejects_malformed_trajectories() -> None:
    gt = _trajectory()
    frozen = np.repeat(np.eye(4)[None], len(gt), axis=0)
    gate = validate_trajectory_pair(frozen, gt)
    assert gate.ok

    malformed = gt.copy()
    malformed[0, 3, 3] = 2.0
    gate = validate_trajectory_pair(malformed, gt)
    assert not gate.ok
    assert "predicted_invalid_homogeneous_row" in gate.errors


def test_frozen_candidate_is_not_an_automatic_dnf(tmp_path):
    gt = _trajectory()
    frozen = np.repeat(np.eye(4)[None], len(gt), axis=0)
    artifact = write_tracking_review_artifact(
        frozen, gt, tmp_path, dataset="d", sequence="s", system="frozen", config_hash="c"
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["gate"]["ok"]
    assert manifest["gate_policy"] == "structural_only_v2"
    assert manifest["files"]["diagnostic"] is None
    assert "predicted trajectory" in manifest["diagnostic_unavailable_reason"]


def test_invalid_reference_cannot_generate_candidate_failure_vote(tmp_path):
    gt = _trajectory()
    gt[0, 3, 3] = 2
    artifacts = [
        write_tracking_review_artifact(
            _trajectory(), gt, tmp_path, dataset="d", sequence="s", system=name, config_hash="c"
        )
        for name in ("a", "b")
    ]
    with pytest.raises(ValueError, match="reference is invalid"):
        build_review_pair(*artifacts, tmp_path / "pair", task_id="s", comparison_id="bad-ref")


def test_valid_artifact_contains_manifest_poses_metrics_and_diagnostic(tmp_path) -> None:
    gt = _trajectory()
    pred = gt.copy()
    pred[10:, 0, 3] += 0.1
    artifact = write_tracking_review_artifact(
        pred,
        gt,
        tmp_path,
        dataset="tum/dynamics",
        sequence="walking/xyz",
        system="method A",
        config_hash="abc123",
        metrics={"trajectory_ate_rmse": 0.02},
    )
    assert (artifact / "poses.npz").is_file()
    assert (artifact / "metrics.json").is_file()
    assert (artifact / "diagnostic.png").is_file()
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["gate"]["ok"] is True
    assert manifest["dataset"] == "tum/dynamics"
    assert manifest["files"]["diagnostic"] == "diagnostic.png"
    metrics = json.loads((artifact / "metrics.json").read_text())
    assert metrics["trajectory_ate_rmse"] == 0.02
    assert "review_rotation_error_deg_p95" in metrics


def test_failed_artifact_is_explicit_and_has_no_diagnostic(tmp_path) -> None:
    gt = _trajectory()
    pred = gt.copy()
    pred[3] = np.nan
    artifact = write_tracking_review_artifact(
        pred,
        gt,
        tmp_path,
        dataset="d",
        sequence="s",
        system="m",
        config_hash="c",
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["gate"]["ok"] is False
    assert "nonfinite_pose_rendering_unsupported" in manifest["gate"]["errors"]
    assert manifest["files"]["diagnostic"] is None
    assert not (artifact / "diagnostic.png").exists()


def test_blinded_pair_and_vote_round_trip(tmp_path) -> None:
    gt = _trajectory()
    first = write_tracking_review_artifact(
        gt,
        gt,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="first",
        config_hash="a",
    )
    second_pred = gt.copy()
    second_pred[10:, 0, 3] += 0.2
    second = write_tracking_review_artifact(
        second_pred,
        gt,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="second",
        config_hash="b",
    )
    pair_path = build_review_pair(
        first,
        second,
        tmp_path / "pair",
        task_id="tum/walk",
        comparison_id="comparison-1",
    )
    pair = json.loads(pair_path.read_text())
    assert pair["system_a"] != pair["system_b"]
    assert pair["automatic_winner"] is None
    assert (pair_path.parent / "review.png").is_file()

    log = tmp_path / "comparisons.jsonl"
    row = record_review_vote(pair_path, "a", log, reviewer="human-1")
    assert row["source"] == "human"
    assert json.loads(log.read_text())["comparison_id"] == "comparison-1"
    with pytest.raises(ValueError, match="already recorded"):
        record_review_vote(pair_path, "b", log, reviewer="human-1")
    record_review_vote(pair_path, "b", log, reviewer="human-2")
    assert len(log.read_text().splitlines()) == 2


def test_pair_rejects_different_reference_frames(tmp_path) -> None:
    gt = _trajectory()
    first = write_tracking_review_artifact(
        gt, gt, tmp_path / "artifacts", dataset="d", sequence="s", system="a", config_hash="1"
    )
    changed = gt.copy()
    changed[5, 0, 3] += 0.01
    second = write_tracking_review_artifact(
        changed,
        changed,
        tmp_path / "artifacts",
        dataset="d",
        sequence="s",
        system="b",
        config_hash="2",
    )
    with pytest.raises(ValueError, match="reference frames"):
        build_review_pair(first, second, tmp_path / "pair", task_id="s", comparison_id="x")


def test_calibration_counts_missing_judge_rows_as_uncovered() -> None:
    human = [{"comparison_id": str(i), "winner": "a", "source": "human"} for i in range(10)]
    judge = [{"comparison_id": "0", "winner": "a", "source": "vlm"}]
    result = calibrate_judge_votes(human, judge)
    assert result["accuracy"] == 1.0
    assert result["coverage"] == 0.1
    assert result["n_human_decisive"] == 10
    with pytest.raises(ValueError, match="aggregate"):
        calibrate_judge_votes(human + human[:1], judge)


def test_calibration_resamples_whole_tasks() -> None:
    human = [{"comparison_id": str(i), "task_id": str(i // 3), "winner": "a"} for i in range(6)]
    judge = [{**r, "winner": "a" if r["task_id"] == "0" else "b"} for r in human]
    result = calibrate_judge_votes(human, judge, n_bootstrap=500, seed=12)
    assert result == calibrate_judge_votes(human, judge, n_bootstrap=500, seed=12)
    uncertainty = result["task_bootstrap"]
    assert uncertainty["n_tasks"] == 2
    assert uncertainty["accuracy_ci95"] == [0.0, 1.0]
    assert uncertainty["coverage_ci95"] == [1.0, 1.0]
    assert uncertainty["accuracy_valid_draws"] == 500


def test_calibration_does_not_invent_independent_tasks() -> None:
    rows = [{"comparison_id": "p", "winner": "a"}]
    assert calibrate_judge_votes(rows, rows)["task_bootstrap"]["accuracy_ci95"] is None
    rows[0]["task_id"] = "one"
    assert calibrate_judge_votes(rows, rows)["task_bootstrap"]["accuracy_ci95"] is None
    with pytest.raises(ValueError, match="task_id mismatch"):
        calibrate_judge_votes(rows, [{**rows[0], "task_id": "two"}])
    with pytest.raises(ValueError, match="invalid calibration verdict"):
        calibrate_judge_votes(rows, [{**rows[0], "winner": "typo"}])


def test_calibration_abstaining_bootstrap_draws_are_reported() -> None:
    human = [{"comparison_id": str(i), "task_id": str(i), "winner": "a"} for i in range(2)]
    result = calibrate_judge_votes(human, human[:1], n_bootstrap=500)
    uncertainty = result["task_bootstrap"]
    assert uncertainty["coverage_ci95"] == [0.0, 1.0]
    assert 0 < uncertainty["accuracy_valid_draws"] < 500
    assert uncertainty["accuracy_ci95"] == [1.0, 1.0]


def test_gated_pair_has_automatic_outcome(tmp_path) -> None:
    gt = _trajectory()
    valid = write_tracking_review_artifact(
        gt,
        gt,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="valid",
        config_hash="a",
    )
    invalid_pred = gt.copy()
    invalid_pred[0] = np.nan
    invalid = write_tracking_review_artifact(
        invalid_pred,
        gt,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="invalid",
        config_hash="b",
    )
    pair_path = build_review_pair(
        valid,
        invalid,
        tmp_path / "pair",
        task_id="tum/walk",
        comparison_id="comparison-2",
    )
    pair = json.loads(pair_path.read_text())
    assert pair["automatic_winner"] in {"a", "b"}
    assert pair["review_image"] is None
    row = record_review_vote(
        pair_path,
        pair["automatic_winner"],
        tmp_path / "comparisons.jsonl",
    )
    assert row["source"] == "gate"


def test_judge_calibration_reports_coverage_accuracy_and_kappa() -> None:
    human = [
        {"comparison_id": "1", "winner": "a", "source": "human"},
        {"comparison_id": "2", "winner": "b", "source": "human"},
        {"comparison_id": "3", "winner": "a", "source": "human"},
        {"comparison_id": "4", "winner": "a", "source": "gate"},
    ]
    judge = [
        {"comparison_id": "1", "winner": "a", "source": "vlm"},
        {"comparison_id": "2", "winner": "a", "source": "vlm"},
        {"comparison_id": "3", "winner": "tie", "source": "vlm"},
        {"comparison_id": "4", "winner": "a", "source": "gate"},
    ]
    result = calibrate_judge_votes(human, judge)
    assert result["n_shared"] == 3
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["cohen_kappa"] == pytest.approx(0.0)
    assert result["confusion"] == {"a": {"a": 1, "b": 0}, "b": {"a": 1, "b": 0}}


def test_provider_neutral_judge_receives_image_and_returns_verdict(tmp_path) -> None:
    gt = _trajectory()
    first = write_tracking_review_artifact(
        gt, gt, tmp_path / "artifacts", dataset="tum", sequence="walk", system="x", config_hash="1"
    )
    changed = gt.copy()
    changed[:, 1, 3] += np.linspace(0, 0.2, len(gt))
    second = write_tracking_review_artifact(
        changed,
        gt,
        tmp_path / "artifacts",
        dataset="tum",
        sequence="walk",
        system="y",
        config_hash="2",
    )
    pair = build_review_pair(
        first, second, tmp_path / "pair", task_id="tum/walk", comparison_id="judge-1"
    )
    adapter = tmp_path / "judge.py"
    adapter.write_text(
        "import json, pathlib, sys\n"
        "request=json.load(sys.stdin)\n"
        "assert pathlib.Path(request['review_image']).is_file()\n"
        "print(json.dumps({'winner':'a','reason':'less drift','provider':'test-vlm'}))\n"
    )
    result = run_external_judge(pair, [sys.executable, str(adapter)])
    assert result["winner"] == "a"
    assert result["provider"] == "test-vlm"
    assert result["comparison_id"] == "judge-1"
