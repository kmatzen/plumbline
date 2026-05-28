"""Static gate: every reproduction's ``paper_reference.primary_metric`` must be
a metric key its run can actually emit.

If it isn't, ``reproduce.py`` does ``aggregate_metrics.get(primary_metric, nan)``
and the reproduction **silently reports NaN** (or, worse, a wrong-but-plausible
number if the typo resolves to a different key). This is the pi3-dtu 'chamfer'
vs scene-path 'overall' bug class. This test catches it at merge time — no GPU,
no data, no model weights — so a mis-keyed yaml can never reach a paper-match
cell. Keep ``expected_metric_keys`` in sync with ``runner.evaluate``.
"""

from __future__ import annotations

import pytest
import yaml

from plumbline.protocols import apply_protocol
from plumbline.reproduce import REPRODUCTIONS_DIR, expected_metric_keys

_REPRO_YAMLS = sorted(p for p in REPRODUCTIONS_DIR.glob("*.yaml") if p.name != "gpu_queue.yaml")


def test_repro_yamls_discovered() -> None:
    # Guard against a glob that silently matches nothing (which would make
    # every parametrized case vacuously pass).
    assert _REPRO_YAMLS, f"no reproduction yamls found under {REPRODUCTIONS_DIR}"


def test_gate_rejects_the_historical_bug() -> None:
    # The exact pi3-dtu shape: scene-aggregation emits 'overall', never
    # 'chamfer'. Proves expected_metric_keys is not vacuously permissive.
    scene_cfg = {"tasks": ["mvs_depth"], "aggregation": "scene"}
    assert "chamfer" not in expected_metric_keys(scene_cfg)
    assert "overall" in expected_metric_keys(scene_cfg)


@pytest.mark.parametrize("path", _REPRO_YAMLS, ids=lambda p: p.stem)
def test_primary_metric_is_emittable(path) -> None:
    cfg = apply_protocol(yaml.safe_load(path.read_text(encoding="utf-8")))
    pm = (cfg.get("paper_reference") or {}).get("primary_metric")
    assert pm is not None, (
        f"{path.name}: paper_reference.primary_metric is missing — reproduce.py "
        f"would report an arbitrary first metric (next(iter(...)))."
    )
    emittable = expected_metric_keys(cfg)
    assert pm in emittable, (
        f"{path.name}: primary_metric {pm!r} is not emittable by this config "
        f"(tasks={cfg.get('tasks')}, aggregation={cfg.get('aggregation', 'sample')}). "
        f"Run would silently report NaN. Emittable keys: {sorted(emittable)}."
    )


# Loosest legitimate paper-match gate in use is 0.10 (monst3r-sintel-pose ATE,
# a stochastic global-alignment pipeline). A tolerance looser than this lets a
# documented mismatch still register as a ✅ paper-match — exactly what the
# vggt-eth3d cell's old 0.30 did (it made the harness report paper_match=True
# for a +23% MISMATCH; fixed 2026-05-28). Anything above the cap should be a
# deliberate, justified bump here.
_MAX_SANE_TOLERANCE = 0.10


@pytest.mark.parametrize("path", _REPRO_YAMLS, ids=lambda p: p.stem)
def test_tolerance_is_not_absurdly_loose(path) -> None:
    paper = yaml.safe_load(path.read_text(encoding="utf-8")).get("paper_reference") or {}
    tol = paper.get("tolerance_relative")
    if tol is None or paper.get("value") is None:
        return  # informational cell — no paper-match is computed
    assert float(tol) <= _MAX_SANE_TOLERANCE, (
        f"{path.name}: tolerance_relative={tol} exceeds the sane cap "
        f"{_MAX_SANE_TOLERANCE} — a gate this loose lets a real mismatch register "
        f"as a ✅ paper-match. If genuinely needed, bump _MAX_SANE_TOLERANCE with a reason."
    )
