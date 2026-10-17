"""Pairwise review and ranking utilities for camera-tracking evaluations.

The scalar pose metrics remain useful, but they do not encode every quality a
human notices (jitter, catastrophic local failures, loop closure, or a visually
implausible path that happens to align globally). This module is the statistical
core for a review harness that compares diagnostic artifacts pairwise.

The design follows the strongest reusable ideas in 3DCodeBench/4DCodeBench:
balanced A/B presentation, resumable per-task comparisons, Bradley--Terry
ratings, and bootstrap intervals that resample tasks rather than individual
votes. No external benchmark code is copied here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

__all__ = [
    "Comparison",
    "Rating",
    "bootstrap_ratings",
    "choose_balanced_pairs",
    "fit_bradley_terry",
    "render_trajectory_diagnostic",
    "trajectory_diagnostics",
]


@dataclass(frozen=True)
class Comparison:
    """One blinded preference on one sequence/task."""

    task_id: str
    system_a: str
    system_b: str
    winner: str

    def __post_init__(self) -> None:
        if self.system_a == self.system_b:
            raise ValueError("a comparison needs two distinct systems")
        if self.winner not in {"a", "b", "tie"}:
            raise ValueError("winner must be 'a', 'b', or 'tie'")


@dataclass(frozen=True)
class Rating:
    """Centered Bradley--Terry/Elo-style rating with a bootstrap interval."""

    score: float
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Sim(3)-aligned paths and per-frame errors used by review artifacts."""

    gt_positions: NDArray[np.float64]
    pred_positions: NDArray[np.float64]
    translation_error_normalized: NDArray[np.float64]
    rotation_error_deg: NDArray[np.float64]
    reference_radius: float


def _systems(comparisons: list[Comparison]) -> list[str]:
    return sorted({c.system_a for c in comparisons} | {c.system_b for c in comparisons})


def fit_bradley_terry(
    comparisons: list[Comparison], *, max_iter: int = 10_000, tolerance: float = 1e-10
) -> dict[str, float]:
    """Fit centered Elo-style scores using Bradley--Terry iterative scaling.

    Ties contribute half a win to each system. Scores are ``400*log10(skill)``
    and centered at zero, so only differences are meaningful. Every system must
    be connected to the comparison graph; otherwise the relative scale between
    disconnected components is unknowable and a ``ValueError`` is raised.
    """
    names = _systems(comparisons)
    if len(names) < 2:
        raise ValueError("at least two compared systems are required")
    index = {name: i for i, name in enumerate(names)}
    adjacency: list[set[int]] = [set() for _ in names]

    for comparison in comparisons:
        a, b = index[comparison.system_a], index[comparison.system_b]
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    if len(seen) != len(names):
        raise ValueError("comparison graph is disconnected")

    outcomes = np.asarray(
        [1.0 if c.winner == "a" else 0.0 if c.winner == "b" else 0.5 for c in comparisons]
    )
    left = np.asarray([index[c.system_a] for c in comparisons], dtype=np.int64)
    right = np.asarray([index[c.system_b] for c in comparisons], dtype=np.int64)

    def objective(raw: np.ndarray) -> float:
        centered = raw - raw.mean()
        logits = centered[left] - centered[right]
        # log(sigmoid(x)) and log(1-sigmoid(x)) in stable form.
        loss = np.logaddexp(0.0, -logits) * outcomes
        loss += np.logaddexp(0.0, logits) * (1.0 - outcomes)
        return float(loss.sum() + 1e-4 * np.square(centered).sum())

    result = minimize(
        objective,
        np.zeros(len(names), dtype=np.float64),
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tolerance},
    )
    if not np.isfinite(result.fun):
        raise ValueError("Bradley-Terry fit did not converge to a finite solution")
    log_skills = np.asarray(result.x) - np.mean(result.x)
    scores = (400.0 / np.log(10.0)) * log_skills
    scores -= scores.mean()
    return {name: float(scores[index[name]]) for name in names}


def bootstrap_ratings(
    comparisons: list[Comparison], *, iterations: int = 1_000, seed: int = 0
) -> dict[str, Rating]:
    """Return ratings and 95% CIs by resampling whole tasks with replacement."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    point = fit_bradley_terry(comparisons)
    tasks = sorted({c.task_id for c in comparisons})
    by_task = {task: [c for c in comparisons if c.task_id == task] for task in tasks}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}

    for _ in range(iterations):
        sampled: list[Comparison] = []
        for draw_index in rng.integers(0, len(tasks), size=len(tasks)):
            sampled.extend(by_task[tasks[int(draw_index)]])
        try:
            fitted = fit_bradley_terry(sampled)
        except ValueError:
            # A small bootstrap can disconnect a sparse design. Dropping that
            # draw is more honest than inventing a cross-component ordering.
            continue
        for name, score in fitted.items():
            draws[name].append(score)

    return {
        name: Rating(
            score=score,
            ci_low=float(np.percentile(draws[name], 2.5)) if draws[name] else None,
            ci_high=float(np.percentile(draws[name], 97.5)) if draws[name] else None,
        )
        for name, score in point.items()
    }


def choose_balanced_pairs(
    systems: list[str], existing: list[Comparison], *, count: int
) -> list[tuple[str, str]]:
    """Choose the least-observed system pairs, deterministically.

    This is suitable for one task/sequence at a time. It balances global pair
    counts while using lexical order to make interrupted/resumed runs stable.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    unique = sorted(set(systems))
    pair_counts: Counter[tuple[str, str]] = Counter()
    for comparison in existing:
        left, right = sorted((comparison.system_a, comparison.system_b))
        pair_counts[(left, right)] += 1
    candidates = list(combinations(unique, 2))
    candidates.sort(key=lambda pair: (pair_counts[pair], pair))
    return candidates[:count]


def _validate_trajectory(extrinsics: NDArray[Any], name: str) -> NDArray[np.float64]:
    array = np.asarray(extrinsics, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4) or len(array) < 2:
        raise ValueError(f"{name} must have shape (N,4,4) with N>=2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _sim3_align(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Umeyama similarity mapping ``source`` points onto ``target`` points."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("predicted trajectory has no translational extent")
    scale = float(np.sum(singular * sign) / variance)
    offset = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, offset


def trajectory_diagnostics(
    predicted: NDArray[Any], reference: NDArray[Any]
) -> TrajectoryDiagnostics:
    """Align two ``world_from_camera`` trajectories and derive review signals."""
    pred = _validate_trajectory(predicted, "predicted")
    gt = _validate_trajectory(reference, "reference")
    if len(pred) != len(gt):
        raise ValueError("predicted and reference trajectories must have equal frame counts")
    gt_positions = gt[:, :3, 3]
    pred_positions = pred[:, :3, 3]
    scale, alignment_rotation, offset = _sim3_align(pred_positions, gt_positions)
    aligned_positions = scale * (pred_positions @ alignment_rotation.T) + offset

    centered_gt = gt_positions - gt_positions.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered_gt**2, axis=1))))
    radius = max(radius, np.finfo(np.float64).eps)
    translation_error = np.linalg.norm(aligned_positions - gt_positions, axis=1) / radius

    aligned_rotations = alignment_rotation[None] @ pred[:, :3, :3]
    relative = np.swapaxes(gt[:, :3, :3], 1, 2) @ aligned_rotations
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    cosine[np.isclose(cosine, 1.0, rtol=0.0, atol=1e-12)] = 1.0
    cosine[np.isclose(cosine, -1.0, rtol=0.0, atol=1e-12)] = -1.0
    rotation_error = np.rad2deg(np.arccos(cosine))
    return TrajectoryDiagnostics(
        gt_positions=gt_positions,
        pred_positions=aligned_positions,
        translation_error_normalized=translation_error,
        rotation_error_deg=rotation_error,
        reference_radius=radius,
    )


def render_trajectory_diagnostic(
    predicted: NDArray[Any],
    reference: NDArray[Any],
    output: str | Path,
    *,
    width: int = 1200,
    height: int = 720,
) -> Path:
    """Render a deterministic four-panel trajectory review PNG.

    Spatial plot bounds come only from GT, preventing each candidate from
    receiving a flattering auto-scale. Error panels use fixed normalized caps
    (one GT trajectory radius and 45 degrees); clipped counts remain visible.
    """
    if width < 600 or height < 400:
        raise ValueError("diagnostic canvas must be at least 600x400")
    diagnostics = trajectory_diagnostics(predicted, reference)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:  # pragma: no cover - Pillow normally bundles this font
        font = ImageFont.load_default()
    margin, gap = 36, 28
    panel_width = (width - 2 * margin - gap) // 2
    panel_height = (height - 2 * margin - gap) // 2
    panels = [
        (margin, margin, margin + panel_width, margin + panel_height),
        (margin + panel_width + gap, margin, width - margin, margin + panel_height),
        (margin, margin + panel_height + gap, margin + panel_width, height - margin),
        (
            margin + panel_width + gap,
            margin + panel_height + gap,
            width - margin,
            height - margin,
        ),
    ]

    _draw_path_panel(draw, panels[0], diagnostics, axes=(0, 2), title="top: X / Z", font=font)
    _draw_path_panel(draw, panels[1], diagnostics, axes=(2, 1), title="side: Z / Y", font=font)
    _draw_series_panel(
        draw,
        panels[2],
        diagnostics.translation_error_normalized,
        cap=1.0,
        title="translation error / GT radius",
        color=(188, 55, 55),
        font=font,
    )
    _draw_series_panel(
        draw,
        panels[3],
        diagnostics.rotation_error_deg,
        cap=45.0,
        title="rotation error (degrees)",
        color=(126, 70, 170),
        font=font,
    )
    draw.text(
        (margin, height - 22),
        "GT blue | candidate orange | Sim(3) aligned",
        fill="black",
        font=font,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return path


def _draw_frame(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, font: Any
) -> None:
    draw.rectangle(box, outline=(80, 80, 80), width=1)
    draw.text((box[0] + 8, box[1] + 6), title, fill="black", font=font)


def _draw_path_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    diagnostics: TrajectoryDiagnostics,
    *,
    axes: tuple[int, int],
    title: str,
    font: Any,
) -> None:
    _draw_frame(draw, box, title, font)
    gt = diagnostics.gt_positions[:, axes]
    pred = diagnostics.pred_positions[:, axes]
    low = gt.min(axis=0)
    high = gt.max(axis=0)
    span = np.maximum(high - low, diagnostics.reference_radius * 0.05)
    low -= 0.1 * span
    high += 0.1 * span
    inner = (box[0] + 12, box[1] + 32, box[2] - 12, box[3] - 12)

    def pixels(points: NDArray[np.float64]) -> list[tuple[int, int]]:
        normalized = np.clip((points - low) / (high - low), 0.0, 1.0)
        return [
            (
                int(inner[0] + p[0] * (inner[2] - inner[0])),
                int(inner[3] - p[1] * (inner[3] - inner[1])),
            )
            for p in normalized
        ]

    draw.line(pixels(gt), fill=(40, 95, 190), width=3)
    draw.line(pixels(pred), fill=(225, 125, 35), width=3)


def _draw_series_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: NDArray[np.float64],
    *,
    cap: float,
    title: str,
    color: tuple[int, int, int],
    font: Any,
) -> None:
    clipped_count = int(np.count_nonzero(values > cap))
    _draw_frame(draw, box, f"{title} | clipped>{cap:g}: {clipped_count}", font)
    inner = (box[0] + 12, box[1] + 32, box[2] - 12, box[3] - 18)
    draw.line((inner[0], inner[3], inner[2], inner[3]), fill=(170, 170, 170), width=1)
    denom = max(len(values) - 1, 1)
    points = [
        (
            int(inner[0] + i / denom * (inner[2] - inner[0])),
            int(inner[3] - min(float(value), cap) / cap * (inner[3] - inner[1])),
        )
        for i, value in enumerate(values)
    ]
    draw.line(points, fill=color, width=2)
