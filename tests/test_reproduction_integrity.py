"""Static gate: every reproduction's model / protocol / dataset references must
resolve to a registered component, and every ``verified_pdf`` value must carry a
citation.

A typo'd ``protocol:``, ``model.name``, or ``dataset.name`` otherwise survives
CI (the slug still resolves to a file) and only blows up on a GPU box after the
data is staged — wasting a run. Same spirit as ``test_reproduction_metric_keys``:
catch it at merge time, no GPU/data/weights. Complements
``test_every_bundled_yaml_is_invokable_by_its_name`` (which only checks the slug
resolves), by validating what the slug points *at*.
"""

from __future__ import annotations

import pytest
import yaml

from plumbline._discover import register_builtin_adapters
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.protocols import apply_protocol
from plumbline.reproduce import REPRODUCTIONS_DIR

# Populate the built-in registries (idempotent; missing optional deps are soft).
register_builtin_adapters()

_REPRO_YAMLS = sorted(p for p in REPRODUCTIONS_DIR.glob("*.yaml") if p.name != "gpu_queue.yaml")


def _is_reproduction(cfg) -> bool:
    return isinstance(cfg, dict) and "model" in cfg


def test_registries_populated_and_not_vacuous() -> None:
    # If the registries were empty the parametrized resolve-checks would be
    # asserting against nothing meaningful; pin that they're populated and that
    # a deliberately-bogus name does NOT resolve (the assertions have teeth).
    assert len(MODEL_REGISTRY) >= 10 and len(DATASET_REGISTRY) >= 10
    assert "definitely-not-a-real-model" not in MODEL_REGISTRY
    assert "definitely-not-a-real-dataset" not in DATASET_REGISTRY


def test_some_reproductions_discovered() -> None:
    assert len(_REPRO_YAMLS) >= 20, f"too few reproduction yamls under {REPRODUCTIONS_DIR}"


@pytest.mark.parametrize("path", _REPRO_YAMLS, ids=lambda p: p.stem)
def test_references_resolve(path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not _is_reproduction(raw):
        pytest.skip(f"{path.name} is not a reproduction (no model: block)")

    # apply_protocol raises FileNotFoundError if `protocol:` names a missing
    # preset, and merges the protocol's fixed.dataset into cfg["dataset"].
    cfg = apply_protocol(raw)

    model_name = (cfg.get("model") or {}).get("name")
    assert model_name in MODEL_REGISTRY, (
        f"{path.name}: model {model_name!r} is not a registered adapter "
        f"(register it in plumbline/_discover.py or fix the typo)."
    )

    dataset_name = (cfg.get("dataset") or {}).get("name")
    assert dataset_name in DATASET_REGISTRY, (
        f"{path.name}: dataset {dataset_name!r} is not a registered loader "
        f"(register it in plumbline/_discover.py or fix the typo)."
    )


@pytest.mark.parametrize("path", _REPRO_YAMLS, ids=lambda p: p.stem)
def test_verified_pdf_has_citation(path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not _is_reproduction(raw):
        pytest.skip(f"{path.name} is not a reproduction (no model: block)")
    pr = raw.get("paper_reference") or {}
    if pr.get("source_confidence") == "verified_pdf" and pr.get("value") is not None:
        assert pr.get("citation"), (
            f"{path.name}: source_confidence=verified_pdf with a pinned value but "
            f"no citation — a verified cell must record the table/column/row it "
            f"was checked against."
        )
