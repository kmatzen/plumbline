"""Small, dependency-light Bayesian design search for expensive evaluations.

The search works over an explicitly declared finite/continuous parameter space.
It fits Gaussian-process surrogates to completed observations and proposes one
new configuration.  Repeated calls use seeded random scalarizations of all
objectives (the ParEGO pattern), so no permanent weighted sum is hidden in the
tool.  Constraint surrogates reduce acquisition value where feasibility is
unlikely.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

__all__ = ["SearchProposal", "propose_configuration"]


@dataclass(frozen=True)
class SearchProposal:
    """One reproducible candidate returned by the design search."""

    parameters: dict[str, object]
    config_hash: str
    acquisition: float
    strategy: str
    objective_weights: dict[str, float]
    n_observations: int
    seed: int


def _parameter_names(spec: dict[str, Any]) -> list[str]:
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("search spec needs a non-empty 'parameters' object")
    return sorted(parameters)


def _validate_spec(
    spec: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    names = _parameter_names(spec)
    parameters = spec["parameters"]
    for name in names:
        item = parameters[name]
        if not isinstance(item, dict):
            raise ValueError(f"parameter {name!r} must be an object")
        kind = item.get("type")
        if kind == "choice":
            values = item.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"choice parameter {name!r} needs non-empty values")
            if len({json.dumps(v, sort_keys=True) for v in values}) != len(values):
                raise ValueError(f"choice parameter {name!r} has duplicate values")
        elif kind in {"float", "int"}:
            low, high = item.get("low"), item.get("high")
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(f"numeric parameter {name!r} needs low and high")
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"parameter {name!r} needs finite low < high")
            if item.get("log", False) and low <= 0:
                raise ValueError(f"log parameter {name!r} needs low > 0")
        else:
            raise ValueError(f"parameter {name!r} has unknown type {kind!r}")

    objectives = spec.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("search spec needs a non-empty 'objectives' list")
    for objective in objectives:
        if not isinstance(objective, dict) or not isinstance(objective.get("name"), str):
            raise ValueError("each objective needs a name")
        if objective.get("direction") not in {"maximize", "minimize"}:
            raise ValueError("objective direction must be 'maximize' or 'minimize'")

    constraints = spec.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    for constraint in constraints:
        if (
            not isinstance(constraint, dict)
            or not isinstance(constraint.get("name"), str)
            or constraint.get("op") not in {">=", "<="}
            or not isinstance(constraint.get("value"), (int, float))
        ):
            raise ValueError("each constraint needs name, op (>= or <=), and numeric value")
    return names, objectives, constraints


def _sample_candidates(
    spec: dict[str, Any], names: list[str], *, count: int, rng: np.random.Generator
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    attempts = max(count * 10, 100)
    for _ in range(attempts):
        candidate: dict[str, object] = {}
        for name in names:
            item = spec["parameters"][name]
            kind = item["type"]
            if kind == "choice":
                values = item["values"]
                candidate[name] = values[int(rng.integers(0, len(values)))]
            elif kind == "int":
                int_low, int_high = int(item["low"]), int(item["high"])
                if item.get("log", False):
                    raw = np.exp(rng.uniform(np.log(int_low), np.log(int_high + 1)))
                    candidate[name] = int(np.clip(np.floor(raw), int_low, int_high))
                else:
                    candidate[name] = int(rng.integers(int_low, int_high + 1))
            else:
                float_low, float_high = float(item["low"]), float(item["high"])
                if item.get("log", False):
                    candidate[name] = float(
                        np.exp(rng.uniform(np.log(float_low), np.log(float_high)))
                    )
                else:
                    candidate[name] = float(rng.uniform(float_low, float_high))
        signature = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            candidates.append(candidate)
        if len(candidates) >= count:
            break
    if not candidates:
        raise ValueError("parameter space produced no candidates")
    return candidates


def _encode_one(spec: dict[str, Any], names: list[str], values: dict[str, Any]) -> list[float]:
    encoded: list[float] = []
    for name in names:
        if name not in values:
            raise ValueError(f"observation is missing parameter {name!r}")
        item, value = spec["parameters"][name], values[name]
        if item["type"] == "choice":
            choices = item["values"]
            if value not in choices:
                raise ValueError(f"unknown value for parameter {name!r}: {value!r}")
            encoded.extend(float(value == choice) for choice in choices)
        else:
            low, high = float(item["low"]), float(item["high"])
            numeric = float(value)
            if not low <= numeric <= high:
                raise ValueError(f"parameter {name!r} is outside [{low}, {high}]")
            if item.get("log", False):
                numeric, low, high = np.log(numeric), np.log(low), np.log(high)
            encoded.append(float((numeric - low) / (high - low)))
    return encoded


def _gp_predict(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.float64],
    query_x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RBF Gaussian-process posterior with standardized targets."""
    mean_y = float(train_y.mean())
    scale_y = float(train_y.std())
    if scale_y < 1e-9:
        scale_y = 1.0
    y = (train_y - mean_y) / scale_y
    length = max(0.2, 1.0 / np.sqrt(train_x.shape[1]))

    def kernel(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * squared / (length**2))

    covariance = kernel(train_x, train_x)
    covariance.flat[:: len(train_x) + 1] += 1e-6
    try:
        factor = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
        cross = kernel(train_x, query_x)
        mean = cross.T @ alpha
        solved = np.linalg.solve(factor, cross)
        variance = np.maximum(1.0 - np.sum(solved**2, axis=0), 1e-12)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
        cross = kernel(train_x, query_x)
        mean = cross.T @ inverse @ y
        variance = np.maximum(1.0 - np.sum(cross * (inverse @ cross), axis=0), 1e-12)
    return mean * scale_y + mean_y, np.sqrt(variance) * scale_y


def _expected_improvement(
    mean: NDArray[np.float64], std: NDArray[np.float64], best: float
) -> NDArray[np.float64]:
    improvement = mean - best - 0.01
    z = improvement / np.maximum(std, 1e-12)
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return np.maximum(improvement * ndtr(z) + std * density, 0.0)


def propose_configuration(
    spec: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    seed: int = 0,
    candidate_count: int = 2048,
) -> SearchProposal:
    """Propose an unevaluated configuration from a mixed parameter space.

    Each observation has ``parameters`` and ``metrics`` objects. Rows whose
    status is not ``"ok"`` are retained as infeasible evidence but do not train
    the objective model. At least three successful rows activate the GP;
    earlier calls use a space-filling maximin proposal.
    """
    if candidate_count < 16:
        raise ValueError("candidate_count must be at least 16")
    names, objectives, constraints = _validate_spec(spec)
    rng = np.random.default_rng(seed)
    candidates = _sample_candidates(spec, names, count=candidate_count, rng=rng)
    existing = {
        json.dumps(row.get("parameters"), sort_keys=True, separators=(",", ":"))
        for row in observations
        if isinstance(row.get("parameters"), dict)
    }
    candidates = [
        item
        for item in candidates
        if json.dumps(item, sort_keys=True, separators=(",", ":")) not in existing
    ]
    if not candidates:
        raise ValueError("candidate pool contains no unevaluated configurations")
    query_x = np.asarray([_encode_one(spec, names, item) for item in candidates])

    modeled_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for row in observations:
        if not isinstance(row.get("parameters"), dict):
            continue
        _encode_one(spec, names, row["parameters"])
        modeled_rows.append(row)
        metrics = row.get("metrics")
        if row.get("status", "ok") != "ok" or not isinstance(metrics, dict):
            continue
        needed = [item["name"] for item in objectives]
        if all(name in metrics and np.isfinite(float(metrics[name])) for name in needed):
            successful.append(row)

    weights_array = rng.dirichlet(np.ones(len(objectives)))
    weights = {
        str(objective["name"]): float(weight)
        for objective, weight in zip(objectives, weights_array, strict=True)
    }
    if len(successful) < 3:
        if successful:
            train_x = np.asarray(
                [_encode_one(spec, names, row["parameters"]) for row in successful]
            )
            distances = np.sqrt(((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(2))
            acquisition = distances.min(axis=1)
        else:
            acquisition = np.ones(len(candidates))
        strategy = "space_filling"
    else:
        train_x = np.asarray([_encode_one(spec, names, row["parameters"]) for row in successful])
        objective_columns: list[NDArray[np.float64]] = []
        for objective in objectives:
            values = np.asarray([float(row["metrics"][objective["name"]]) for row in successful])
            if objective["direction"] == "minimize":
                values = -values
            spread = float(np.ptp(values))
            objective_columns.append((values - values.min()) / (spread if spread > 1e-12 else 1.0))
        normalized = np.column_stack(objective_columns)
        # Seeded random scalarization (ParEGO): repeated proposals explore the
        # Pareto surface rather than baking in one preference vector.
        scalar = normalized @ weights_array
        mean, std = _gp_predict(train_x, scalar, query_x)
        acquisition = _expected_improvement(mean, std, float(scalar.max()))
        strategy = "bayesian_random_scalarization"

        failed = [row for row in modeled_rows if row.get("status", "ok") != "ok"]
        if failed and len(modeled_rows) >= 3:
            status_x = np.asarray(
                [_encode_one(spec, names, row["parameters"]) for row in modeled_rows]
            )
            status_y = np.asarray([float(row.get("status", "ok") == "ok") for row in modeled_rows])
            success_mean, success_std = _gp_predict(status_x, status_y, query_x)
            acquisition *= np.clip(
                ndtr((success_mean - 0.5) / np.maximum(success_std, 1e-12)), 0.0, 1.0
            )

        for constraint in constraints:
            rows = [
                row
                for row in successful
                if constraint["name"] in row["metrics"]
                and np.isfinite(float(row["metrics"][constraint["name"]]))
            ]
            if len(rows) < 3:
                continue
            constraint_x = np.asarray([_encode_one(spec, names, row["parameters"]) for row in rows])
            values = np.asarray([float(row["metrics"][constraint["name"]]) for row in rows])
            c_mean, c_std = _gp_predict(constraint_x, values, query_x)
            threshold = float(constraint["value"])
            signed = c_mean - threshold if constraint["op"] == ">=" else threshold - c_mean
            acquisition *= ndtr(signed / np.maximum(c_std, 1e-12))

    index = int(np.argmax(acquisition))
    selected = candidates[index]
    serialized = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return SearchProposal(
        parameters=selected,
        config_hash=hashlib.sha256(serialized.encode()).hexdigest()[:16],
        acquisition=float(acquisition[index]),
        strategy=strategy,
        objective_weights=weights,
        n_observations=len(successful),
        seed=seed,
    )
