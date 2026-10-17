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

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

from plumbline._version import __version__
from plumbline.cache import sanitize_id

__all__ = [
    "Comparison",
    "Rating",
    "TrajectoryGate",
    "bootstrap_ratings",
    "build_review_pair",
    "calibrate_judge_votes",
    "choose_balanced_pairs",
    "fit_bradley_terry",
    "record_review_vote",
    "render_trajectory_diagnostic",
    "trajectory_diagnostics",
    "validate_trajectory_pair",
    "write_tracking_review_artifact",
]

ARTIFACT_SCHEMA_VERSION = "1.0.0"
RENDERER_VERSION = "1"


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
        if self.winner not in {"a", "b", "tie", "both_bad"}:
            raise ValueError("winner must be 'a', 'b', 'tie', or 'both_bad'")


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


@dataclass(frozen=True)
class TrajectoryGate:
    """Deterministic validity decision made before rendering or judging."""

    ok: bool
    errors: tuple[str, ...]
    n_frames: int
    finite_fraction: float


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
    comparisons = [comparison for comparison in comparisons if comparison.winner != "both_bad"]
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
    rankable = [comparison for comparison in comparisons if comparison.winner != "both_bad"]
    point = fit_bradley_terry(rankable)
    tasks = sorted({c.task_id for c in rankable})
    by_task = {task: [c for c in rankable if c.task_id == task] for task in tasks}
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


def validate_trajectory_pair(
    predicted: NDArray[Any], reference: NDArray[Any], *, min_finite_fraction: float = 1.0
) -> TrajectoryGate:
    """Validate a candidate/reference pair without raising.

    The first artifact version requires every pose to be finite by default.
    ``min_finite_fraction`` is exposed for future gap-aware renderers, but a
    non-finite accepted pose is still reported as unsupported rather than being
    silently interpolated.
    """
    if not 0.0 < min_finite_fraction <= 1.0:
        raise ValueError("min_finite_fraction must be in (0, 1]")
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(reference, dtype=np.float64)
    errors: list[str] = []
    if pred.ndim != 3 or pred.shape[1:] != (4, 4):
        errors.append("predicted_shape")
    if gt.ndim != 3 or gt.shape[1:] != (4, 4):
        errors.append("reference_shape")
    if errors:
        return TrajectoryGate(False, tuple(errors), 0, 0.0)
    if len(pred) != len(gt):
        errors.append("frame_count_mismatch")
    if len(pred) < 2 or len(gt) < 2:
        errors.append("too_few_frames")
    n_frames = min(len(pred), len(gt))
    if n_frames == 0:
        return TrajectoryGate(False, tuple(errors), 0, 0.0)

    finite = np.isfinite(pred[:n_frames]).all(axis=(1, 2)) & np.isfinite(gt[:n_frames]).all(
        axis=(1, 2)
    )
    finite_fraction = float(finite.mean())
    if finite_fraction < min_finite_fraction:
        errors.append("insufficient_finite_coverage")
    if not finite.all():
        errors.append("nonfinite_pose_rendering_unsupported")
    if finite.any():
        for name, trajectory in (
            ("predicted", pred[:n_frames][finite]),
            ("reference", gt[:n_frames][finite]),
        ):
            bottom = trajectory[:, 3, :]
            expected = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), bottom.shape)
            if not np.allclose(bottom, expected, atol=1e-6):
                errors.append(f"{name}_invalid_homogeneous_row")
            rotations = trajectory[:, :3, :3]
            identity = np.eye(3)[None]
            if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, identity, atol=1e-3):
                errors.append(f"{name}_non_orthonormal_rotation")
            determinants = np.linalg.det(rotations)
            if not np.allclose(determinants, 1.0, atol=1e-3):
                errors.append(f"{name}_improper_rotation")

    if finite.all() and n_frames >= 2:
        gt_extent = float(np.ptp(gt[:n_frames, :3, 3], axis=0).max())
        pred_extent = float(np.ptp(pred[:n_frames, :3, 3], axis=0).max())
        if gt_extent > 1e-6 and pred_extent <= max(gt_extent * 1e-6, 1e-12):
            errors.append("frozen_predicted_trajectory")
    return TrajectoryGate(not errors, tuple(dict.fromkeys(errors)), n_frames, finite_fraction)


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


def _array_fingerprint(*arrays: NDArray[Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_tracking_review_artifact(
    predicted: NDArray[Any],
    reference: NDArray[Any],
    root: str | Path,
    *,
    dataset: str,
    sequence: str,
    system: str,
    config_hash: str,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Write one stable review artifact directory and return its path.

    Invalid pairs still produce ``poses.npz``, ``metrics.json``, and a failed
    ``manifest.json``. Only a gate-passing pair receives ``diagnostic.png``.
    This makes mechanical failure an explicit evaluation outcome.
    """
    pred = np.asarray(predicted)
    gt = np.asarray(reference)
    artifact_dir = (
        Path(root)
        / sanitize_id(dataset)
        / sanitize_id(sequence)
        / sanitize_id(system)
        / sanitize_id(config_hash)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_dir / "poses.npz", predicted=pred, reference=gt)

    gate = validate_trajectory_pair(pred, gt)
    recorded_metrics = dict(metrics or {})
    diagnostic_path: Path | None = None
    if gate.ok:
        diagnostics = trajectory_diagnostics(pred, gt)
        recorded_metrics.update(
            {
                "review_translation_error_normalized_mean": float(
                    diagnostics.translation_error_normalized.mean()
                ),
                "review_translation_error_normalized_p95": float(
                    np.percentile(diagnostics.translation_error_normalized, 95)
                ),
                "review_rotation_error_deg_mean": float(diagnostics.rotation_error_deg.mean()),
                "review_rotation_error_deg_p95": float(
                    np.percentile(diagnostics.rotation_error_deg, 95)
                ),
            }
        )
        diagnostic_path = render_trajectory_diagnostic(pred, gt, artifact_dir / "diagnostic.png")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(recorded_metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "plumbline_version": __version__,
        "dataset": dataset,
        "sequence": sequence,
        "system": system,
        "config_hash": config_hash,
        "coordinate_convention": "world_from_camera",
        "alignment": "sim3_umeyama",
        "input_fingerprint": _array_fingerprint(pred, gt),
        "gate": asdict(gate),
        "files": {
            "poses": "poses.npz",
            "metrics": "metrics.json",
            "diagnostic": diagnostic_path.name if diagnostic_path else None,
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_dir


def _read_manifest(artifact: Path) -> dict[str, Any]:
    path = artifact / "manifest.json"
    if not path.is_file():
        raise ValueError(f"artifact has no manifest.json: {artifact}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema in {path}")
    return data


def build_review_pair(
    artifact_a: str | Path,
    artifact_b: str | Path,
    output: str | Path,
    *,
    task_id: str,
    comparison_id: str,
) -> Path:
    """Build a deterministic blinded A/B pair bundle and return ``pair.json``."""
    first_path, second_path = Path(artifact_a), Path(artifact_b)
    first, second = _read_manifest(first_path), _read_manifest(second_path)
    if first["system"] == second["system"]:
        raise ValueError("pair requires artifacts from distinct systems")
    expected = (first["dataset"], first["sequence"])
    actual = (second["dataset"], second["sequence"])
    if actual != expected:
        raise ValueError(f"artifacts describe different tasks: {expected!r} != {actual!r}")

    swap = int(hashlib.sha256(comparison_id.encode()).hexdigest()[:8], 16) % 2 == 1
    displayed = (
        [(second_path, second), (first_path, first)]
        if swap
        else [
            (first_path, first),
            (second_path, second),
        ]
    )
    valid = [bool(item[1]["gate"]["ok"]) for item in displayed]
    automatic_winner: str | None = None
    if valid == [True, False]:
        automatic_winner = "a"
    elif valid == [False, True]:
        automatic_winner = "b"
    elif valid == [False, False]:
        automatic_winner = "both_bad"

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_image: str | None = None
    if automatic_winner is None:
        images = [
            Image.open(path / item["files"]["diagnostic"]).convert("RGB")
            for path, item in displayed
        ]
        try:
            panel_width = max(image.width for image in images)
            panel_height = max(image.height for image in images)
            canvas = Image.new("RGB", (panel_width * 2 + 30, panel_height + 54), "white")
            draw = ImageDraw.Draw(canvas)
            font: ImageFont.ImageFont | ImageFont.FreeTypeFont
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except OSError:  # pragma: no cover
                font = ImageFont.load_default()
            for index, image in enumerate(images):
                x = index * (panel_width + 30)
                canvas.paste(image, (x, 54))
                draw.text((x + 16, 14), f"Candidate {'AB'[index]}", fill="black", font=font)
            review_image = "review.png"
            canvas.save(out_dir / review_image, format="PNG", optimize=False)
        finally:
            for image in images:
                image.close()

    pair = {
        "schema_version": "1.0.0",
        "comparison_id": comparison_id,
        "task_id": task_id,
        "dataset": first["dataset"],
        "sequence": first["sequence"],
        "swap": swap,
        "system_a": displayed[0][1]["system"],
        "system_b": displayed[1][1]["system"],
        "artifact_a": str(displayed[0][0].resolve()),
        "artifact_b": str(displayed[1][0].resolve()),
        "review_image": review_image,
        "automatic_winner": automatic_winner,
    }
    pair_path = out_dir / "pair.json"
    pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pair_path


def record_review_vote(
    pair_path: str | Path,
    winner: str,
    comparisons_path: str | Path,
    *,
    reviewer: str | None = None,
    source: str = "human",
) -> dict[str, Any]:
    """Append one non-duplicate pair verdict to the canonical JSONL log."""
    pair_file = Path(pair_path)
    pair = json.loads(pair_file.read_text(encoding="utf-8"))
    if pair.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported pair schema in {pair_file}")
    automatic = pair.get("automatic_winner")
    if automatic is not None:
        if winner != automatic:
            raise ValueError(f"gated pair requires automatic verdict {automatic!r}")
        source = "gate"
    else:
        if winner not in {"a", "b", "tie", "both_bad"}:
            raise ValueError("winner must be 'a', 'b', 'tie', or 'both_bad'")
        if source not in {"human", "vlm"}:
            raise ValueError("source must be 'human' or 'vlm'")

    output = Path(comparisons_path)
    comparison_id = str(pair["comparison_id"])
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("comparison_id") == comparison_id:
                raise ValueError(f"comparison {comparison_id!r} is already recorded")
    row = {
        "comparison_id": comparison_id,
        "task_id": str(pair["task_id"]),
        "system_a": str(pair["system_a"]),
        "system_b": str(pair["system_b"]),
        "winner": winner,
        "source": source,
        "reviewer": reviewer,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def calibrate_judge_votes(
    human_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure a judge log against human labels on shared comparison IDs.

    Accuracy/kappa use only pairs where both sides chose A or B. Coverage is
    judge-decisive / human-decisive, so a judge cannot inflate agreement by
    returning ties on difficult cases. Gate-generated rows are excluded.
    """
    human = {str(row["comparison_id"]): row for row in human_rows if row.get("source") != "gate"}
    judge = {str(row["comparison_id"]): row for row in judge_rows if row.get("source") != "gate"}
    shared_ids = sorted(set(human) & set(judge))
    human_decisive = [cid for cid in shared_ids if human[cid].get("winner") in {"a", "b"}]
    decisive = [cid for cid in human_decisive if judge[cid].get("winner") in {"a", "b"}]
    confusion = {"a": {"a": 0, "b": 0}, "b": {"a": 0, "b": 0}}
    for cid in decisive:
        confusion[str(human[cid]["winner"])][str(judge[cid]["winner"])] += 1
    correct = sum(human[cid]["winner"] == judge[cid]["winner"] for cid in decisive)
    accuracy = correct / len(decisive) if decisive else None
    coverage = len(decisive) / len(human_decisive) if human_decisive else None

    kappa: float | None = None
    if decisive:
        observed = correct / len(decisive)
        human_a = sum(human[cid]["winner"] == "a" for cid in decisive) / len(decisive)
        judge_a = sum(judge[cid]["winner"] == "a" for cid in decisive) / len(decisive)
        expected = human_a * judge_a + (1.0 - human_a) * (1.0 - judge_a)
        if expected < 1.0:
            kappa = (observed - expected) / (1.0 - expected)
    return {
        "n_human": len(human),
        "n_judge": len(judge),
        "n_shared": len(shared_ids),
        "n_human_decisive": len(human_decisive),
        "n_both_decisive": len(decisive),
        "coverage": coverage,
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "confusion": confusion,
    }


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
