#!/usr/bin/env python3
"""Historical exploratory experiments, retained for reproducibility.

The cross-family severity ordering is arbitrary. Its accuracy numbers are
withdrawn and must not be used as evidence of evaluator quality. Use
run_sensitivity.py for the replacement within-family implementation checks.
The design-search experiment is a synthetic optimizer check only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import kendalltau

from plumbline.design_search import propose_configuration
from plumbline.metrics.pose import trajectory_ate_rmse_sim3, trajectory_rpe_rmse_sim3
from plumbline.tracking_review import Comparison, fit_bradley_terry, trajectory_diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "results"
FIGURES = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
SMALL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 23)


def trajectory(seed: int, n: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    phase = rng.uniform(-0.2, 0.2)
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 0, 3] = 1.8 * t
    poses[:, 1, 3] = 0.25 * np.sin(2 * np.pi * (t + phase))
    poses[:, 2, 3] = 0.35 * t**2
    yaw = 0.35 * np.sin(np.pi * t)
    poses[:, 0, 0] = np.cos(yaw)
    poses[:, 0, 2] = np.sin(yaw)
    poses[:, 2, 0] = -np.sin(yaw)
    poses[:, 2, 2] = np.cos(yaw)
    return poses


def perturb(gt: np.ndarray, kind: str, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pred = gt.copy()
    n = len(gt)
    t = np.linspace(0.0, 1.0, n)
    if kind == "drift":
        pred[:, 1, 3] += level * t**2
    elif kind == "jitter":
        pred[:, :3, 3] += rng.normal(0.0, level, size=(n, 3))
    elif kind == "jump":
        width = max(2, n // 12)
        start = n // 2 - width // 2
        pred[start : start + width, 0, 3] += level * 3.2
    elif kind == "rotation":
        angles = rng.normal(0.0, level * 3.0, size=n)
        for i, angle in enumerate(angles):
            c, s = math.cos(angle), math.sin(angle)
            noise = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            pred[i, :3, :3] = noise @ pred[i, :3, :3]
    elif kind != "clean":
        raise ValueError(kind)
    return pred


def review_score(pred: np.ndarray, gt: np.ndarray) -> float:
    diagnostic = trajectory_diagnostics(pred, gt)
    pos = diagnostic.pred_positions
    acceleration = np.diff(pos, n=2, axis=0)
    jerk = np.diff(pos, n=3, axis=0)
    local = float(np.percentile(diagnostic.translation_error_normalized, 95))
    roughness = float(np.mean(np.linalg.norm(acceleration, axis=1)))
    impulse = float(np.max(np.linalg.norm(jerk, axis=1)))
    rotation = float(np.percentile(diagnostic.rotation_error_deg, 95) / 45.0)
    # Fixed before running the comparisons. Higher is better.
    return -(1.0 * local + 2.0 * roughness + 1.5 * impulse + 1.0 * rotation)


def evaluator_experiment() -> dict:
    systems = ["clean", "drift", "jitter", "jump", "rotation"]
    levels = {"clean": 0.0, "drift": 0.10, "jitter": 0.025, "jump": 0.055, "rotation": 0.035}
    expected = {"clean": 4, "drift": 3, "rotation": 2, "jitter": 1, "jump": 0}
    rows = []
    comparisons: list[Comparison] = []
    metric_pair_correct = 0
    review_pair_correct = 0
    pair_total = 0
    for sequence in range(24):
        gt = trajectory(sequence)
        per_system = {}
        for system in systems:
            pred = perturb(gt, system, levels[system], 10_000 + sequence)
            ate = float(trajectory_ate_rmse_sim3(pred, gt))
            rpe_t, rpe_r = trajectory_rpe_rmse_sim3(pred, gt)
            score = review_score(pred, gt)
            per_system[system] = {"ate": ate, "rpe_t": rpe_t, "rpe_r": rpe_r, "score": score}
            rows.append({"sequence": sequence, "system": system, **per_system[system]})
        for i, left in enumerate(systems):
            for right in systems[i + 1 :]:
                if expected[left] == expected[right]:
                    continue
                true_left = expected[left] > expected[right]
                metric_left = per_system[left]["ate"] < per_system[right]["ate"]
                review_left = per_system[left]["score"] > per_system[right]["score"]
                metric_pair_correct += int(metric_left == true_left)
                review_pair_correct += int(review_left == true_left)
                pair_total += 1
                winner = "a" if review_left else "b"
                comparisons.append(Comparison(f"seq-{sequence:02d}", left, right, winner))

    means = {
        system: {
            key: float(np.mean([row[key] for row in rows if row["system"] == system]))
            for key in ("ate", "rpe_t", "rpe_r", "score")
        }
        for system in systems
    }
    ratings = fit_bradley_terry(comparisons)
    expected_vector = [expected[s] for s in systems]
    ate_vector = [-means[s]["ate"] for s in systems]
    score_vector = [means[s]["score"] for s in systems]
    result = {
        "design": {"sequences": 24, "systems": systems, "pairwise_trials": pair_total},
        "means": means,
        "ratings": ratings,
        "ate_pairwise_accuracy": metric_pair_correct / pair_total,
        "review_pairwise_accuracy": review_pair_correct / pair_total,
        "ate_kendall_tau": float(kendalltau(expected_vector, ate_vector).statistic),
        "review_kendall_tau": float(kendalltau(expected_vector, score_vector).statistic),
    }
    (OUT / "controlled_evaluator.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def objective(parameters: dict[str, object]) -> float:
    threshold = float(parameters["threshold"])
    window = int(parameters["window"])
    precision_bonus = 0.025 if parameters["precision"] == "fp32" else 0.0
    peak = math.exp(-((math.log(threshold) - math.log(0.075)) / 0.72) ** 2)
    window_term = math.exp(-((window - 9.0) / 3.2) ** 2)
    interaction = 0.08 * math.sin(window * 0.8 + math.log(threshold) * 1.7)
    return 0.62 * peak + 0.31 * window_term + interaction + precision_bonus


def random_parameters(rng: np.random.Generator) -> dict[str, object]:
    return {
        "threshold": float(np.exp(rng.uniform(np.log(0.001), np.log(0.5)))),
        "window": int(rng.integers(2, 17)),
        "precision": "fp32" if rng.random() > 0.5 else "fp16",
    }


def design_search_experiment() -> dict:
    spec = {
        "parameters": {
            "threshold": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
            "window": {"type": "int", "low": 2, "high": 16},
            "precision": {"type": "choice", "values": ["fp16", "fp32"]},
        },
        "objectives": [{"name": "quality", "direction": "maximize"}],
    }
    grid = [
        objective({"threshold": float(t), "window": w, "precision": p})
        for t in np.geomspace(0.001, 0.5, 500)
        for w in range(2, 17)
        for p in ("fp16", "fp32")
    ]
    optimum = max(grid)
    budgets = list(range(5, 21))
    trials = 60
    bayes_regret = np.zeros((trials, len(budgets)))
    random_regret = np.zeros_like(bayes_regret)
    for trial in range(trials):
        rng = np.random.default_rng(30_000 + trial)
        initial = [random_parameters(rng) for _ in range(4)]
        observations = [
            {"parameters": item, "metrics": {"quality": objective(item)}, "status": "ok"}
            for item in initial
        ]
        random_values = [objective(item) for item in initial]
        for step, budget in enumerate(budgets):
            proposal = propose_configuration(spec, observations, seed=trial * 100 + budget, candidate_count=512)
            value = objective(proposal.parameters)
            observations.append(
                {"parameters": proposal.parameters, "metrics": {"quality": value}, "status": "ok"}
            )
            random_item = random_parameters(rng)
            random_values.append(objective(random_item))
            bayes_regret[trial, step] = optimum - max(row["metrics"]["quality"] for row in observations)
            random_regret[trial, step] = optimum - max(random_values)
    result = {
        "trials": trials,
        "budgets": budgets,
        "optimum": optimum,
        "bayes_median_regret": np.median(bayes_regret, axis=0).tolist(),
        "random_median_regret": np.median(random_regret, axis=0).tolist(),
        "bayes_success_at_20": float(np.mean(bayes_regret[:, -1] <= 0.05)),
        "random_success_at_20": float(np.mean(random_regret[:, -1] <= 0.05)),
    }
    (OUT / "design_search.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def line_plot(evaluator: dict, search: dict) -> None:
    image = Image.new("RGB", (1900, 760), "#f7f4ed")
    draw = ImageDraw.Draw(image)
    draw.text((75, 45), "Controlled validation of the proposed evaluator", fill="#17202a", font=FONT)
    draw.text((75, 100), "Known defect ordering", fill="#46515c", font=SMALL)
    ratings = evaluator["ratings"]
    left, top, width, height = 100, 180, 720, 430
    ordered = ["clean", "drift", "rotation", "jitter", "jump"]
    lo, hi = min(ratings.values()), max(ratings.values())
    for index, system in enumerate(ordered):
        y = top + index * 82
        x = left + 180 + (ratings[system] - lo) / max(hi - lo, 1e-9) * (width - 250)
        draw.line((left + 180, y, left + width, y), fill="#d6d1c7", width=2)
        draw.text((left, y - 14), system, fill="#28323c", font=SMALL)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#146b8c")
    draw.text((left + 180, top + height), "lower preference", fill="#65717d", font=SMALL)
    draw.text((left + width - 170, top + height), "higher", fill="#65717d", font=SMALL)
    draw.text((1040, 100), "Budget-matched design search", fill="#46515c", font=SMALL)
    plot_left, plot_top, plot_w, plot_h = 1030, 180, 760, 430
    draw.line((plot_left, plot_top, plot_left, plot_top + plot_h), fill="#39434d", width=3)
    draw.line((plot_left, plot_top + plot_h, plot_left + plot_w, plot_top + plot_h), fill="#39434d", width=3)
    budgets = search["budgets"]
    values = search["bayes_median_regret"] + search["random_median_regret"]
    max_value = max(values) * 1.05
    for label, key, color in (("Bayesian", "bayes_median_regret", "#146b8c"), ("Random", "random_median_regret", "#c35d2e")):
        points = []
        for budget, value in zip(budgets, search[key], strict=True):
            x = plot_left + (budget - budgets[0]) / (budgets[-1] - budgets[0]) * plot_w
            y = plot_top + plot_h - value / max_value * plot_h
            points.append((x, y))
        draw.line(points, fill=color, width=6)
        draw.text((plot_left + 520, plot_top + (20 if label == "Bayesian" else 62)), label, fill=color, font=SMALL)
    draw.text((plot_left + 250, plot_top + plot_h + 28), "evaluation budget", fill="#65717d", font=SMALL)
    draw.text((plot_left - 10, plot_top - 45), "median regret (lower is better)", fill="#65717d", font=SMALL)
    image.save(FIGURES / "method_validation.png")


if __name__ == "__main__":
    evaluator_result = evaluator_experiment()
    search_result = design_search_experiment()
    line_plot(evaluator_result, search_result)
    print(json.dumps({"evaluator": evaluator_result, "search": search_result}, indent=2))
