from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plumbline.tracking_review import (
    Comparison,
    bootstrap_ratings,
    choose_balanced_pairs,
    fit_bradley_terry,
    render_trajectory_diagnostic,
    trajectory_diagnostics,
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
        Comparison("task", "a", "b", "both_bad")


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


def test_render_trajectory_diagnostic_is_deterministic(tmp_path) -> None:
    gt = _trajectory()
    pred = gt.copy()
    pred[10:, 0, 3] += 0.1
    first = render_trajectory_diagnostic(pred, gt, tmp_path / "first.png")
    second = render_trajectory_diagnostic(pred, gt, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.size == (1200, 720)
