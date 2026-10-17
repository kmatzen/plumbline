"""Measure metric response to increasing corruption within each defect family.

This is a sensitivity test, not a perceptual preference benchmark. No ordering
between different defect families is asserted. All levels share noise draws.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from plumbline.metrics.pose import trajectory_ate_rmse_sim3, trajectory_rpe_rmse_sim3
from plumbline.tracking_review import trajectory_diagnostics


def camera_path(seed: int, frames: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, frames)
    poses = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    poses[:, :3, 3] = np.column_stack(
        (t, rng.uniform(0.15, 0.4) * np.sin(t * 2 * np.pi), t**2 * 0.3)
    )
    return poses


def corrupt(reference: np.ndarray, family: str, level: float, seed: int) -> np.ndarray:
    result = reference.copy()
    n = len(result)
    rng = np.random.default_rng(seed)
    if family == "drift":
        result[:, 1, 3] += level * np.linspace(0, 1, n) ** 2
    elif family == "jitter":
        result[:, :3, 3] += level * rng.normal(size=(n, 3))
    elif family == "jump":
        result[n // 2 : n // 2 + 4, 0, 3] += level
    elif family == "rotation":
        angles = level * rng.normal(size=n)
        result[:, 0, 0] = np.cos(angles)
        result[:, 0, 1] = -np.sin(angles)
        result[:, 1, 0] = np.sin(angles)
        result[:, 1, 1] = np.cos(angles)
    else:
        raise ValueError(family)
    return result


def measurements(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diagnostics = trajectory_diagnostics(prediction, reference)
    rpe_t, rpe_r = trajectory_rpe_rmse_sim3(prediction, reference)
    # Error derivatives remove genuine acceleration in the reference motion.
    residual = diagnostics.pred_positions - diagnostics.gt_positions
    residual /= diagnostics.reference_radius
    return {
        "ate": float(trajectory_ate_rmse_sim3(prediction, reference)),
        "rpe_translation": float(rpe_t),
        "rpe_rotation": float(rpe_r),
        "translation_p95": float(np.percentile(diagnostics.translation_error_normalized, 95)),
        "rotation_p95": float(np.percentile(diagnostics.rotation_error_deg, 95)),
        "residual_acceleration": float(
            np.linalg.norm(np.diff(residual, n=2, axis=0), axis=1).mean()
        ),
        "residual_jerk_max": float(np.linalg.norm(np.diff(residual, n=3, axis=0), axis=1).max()),
    }


def run() -> dict:
    levels = [0.0, 0.005, 0.01, 0.02, 0.04]
    raw = []
    summaries = {}
    for family in ("drift", "jitter", "jump", "rotation"):
        outcomes: dict[str, list[list[float]]] = {}
        for seed in range(24):
            reference = camera_path(seed)
            rows = [
                measurements(corrupt(reference, family, level, seed + 1000), reference)
                for level in levels
            ]
            raw.extend(
                {"family": family, "seed": seed, "level": level, "metrics": row}
                for level, row in zip(levels, rows, strict=True)
            )
            for metric in rows[0]:
                votes = []
                for low, high in combinations(range(len(levels)), 2):
                    delta = rows[high][metric] - rows[low][metric]
                    votes.append(0.5 if abs(delta) <= 1e-8 else float(delta > 0))
                outcomes.setdefault(metric, []).append(votes)
        summaries[family] = {}
        for metric, votes in outcomes.items():
            per_task = np.mean(votes, axis=1)
            rng = np.random.default_rng(103)
            draws = per_task[rng.integers(0, len(per_task), size=(2000, len(per_task)))].mean(
                axis=1
            )
            summaries[family][metric] = {
                "increasing_amplitude_concordance": float(per_task.mean()),
                "task_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "tie_fraction": float(np.mean(np.asarray(votes) == 0.5)),
            }
    return {
        "interpretation": "Amplitude sensitivity only; no perceptual quality labels.",
        "sequences": 24,
        "levels": levels,
        "rows": raw,
        "summary": summaries,
    }


if __name__ == "__main__":
    result = run()
    output = Path(__file__).parent / "results" / "sensitivity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
