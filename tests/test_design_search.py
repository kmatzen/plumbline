from __future__ import annotations

from dataclasses import asdict
from typing import cast

import pytest

from plumbline.design_search import propose_configuration


def _spec() -> dict:
    return {
        "parameters": {
            "precision": {"type": "choice", "values": ["fp16", "fp32"]},
            "threshold": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
            "window": {"type": "int", "low": 2, "high": 12},
        },
        "objectives": [
            {"name": "preference", "direction": "maximize"},
            {"name": "latency", "direction": "minimize"},
        ],
        "constraints": [{"name": "coverage", "op": ">=", "value": 0.9}],
    }


def test_initial_design_is_reproducible_and_in_bounds() -> None:
    first = propose_configuration(_spec(), [], seed=4, candidate_count=128)
    second = propose_configuration(_spec(), [], seed=4, candidate_count=128)
    assert first == second
    assert first.strategy == "space_filling"
    assert first.parameters["precision"] in {"fp16", "fp32"}
    assert 0.01 <= float(first.parameters["threshold"]) <= 1.0
    assert 2 <= cast(int, first.parameters["window"]) <= 12
    assert len(first.config_hash) == 16
    assert sum(first.objective_weights.values()) == pytest.approx(1.0)


def test_bayesian_proposal_excludes_observed_and_uses_all_objectives() -> None:
    observations = []
    for index, window in enumerate((2, 4, 6, 8, 10, 12)):
        observations.append(
            {
                "parameters": {
                    "precision": "fp16" if index % 2 else "fp32",
                    "threshold": 0.02 * (index + 1),
                    "window": window,
                },
                "metrics": {
                    "preference": 0.2 + 0.1 * index,
                    "latency": 30.0 + window,
                    "coverage": 0.8 + 0.03 * index,
                },
                "status": "ok",
            }
        )
    proposal = propose_configuration(_spec(), observations, seed=9, candidate_count=256)
    assert proposal.strategy == "bayesian_random_scalarization"
    assert proposal.n_observations == 6
    assert proposal.acquisition >= 0.0
    assert proposal.parameters not in [row["parameters"] for row in observations]
    assert set(proposal.objective_weights) == {"preference", "latency"}
    assert asdict(proposal)["seed"] == 9


def test_invalid_search_spec_is_rejected() -> None:
    spec = _spec()
    spec["parameters"]["window"]["low"] = 12
    with pytest.raises(ValueError, match="low < high"):
        propose_configuration(spec, [], candidate_count=16)
