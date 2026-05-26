"""Unit tests for the MonST3R adapter's plumbline-side logic.

MonST3R delegates the actual dust3r global alignment to the shared
``_run_mast3r`` runner (covered by the MASt3R tests). What's MonST3R-specific
and worth testing here without a GPU/repo: the single-frame duplicate→slice
path, multi-view pass-through, capability/cap handling, and config_hash.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plumbline.models import monst3r as monst3r_mod  # noqa: E402
from plumbline.models.monst3r import MonST3RAdapter  # noqa: E402


def _fake_run(n: int, h: int = 4, w: int = 5) -> dict[str, Any]:
    """A fake ``_run_mast3r`` return for n views."""
    return {
        "depth": np.full((n, h, w), 2.0, dtype=np.float32),
        "intrinsics": np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1)),
        "extrinsics": np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1)),
        "point_map": np.zeros((n, h, w, 3), dtype=np.float32),
        "confidence": np.ones((n, h, w), dtype=np.float32),
    }


def _patch(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Stub _load (no real model), _run_mast3r (multi-view), and the new
    single-frame helper. Records what each was called with."""

    def fake_load(self: MonST3RAdapter) -> None:
        self._model = object()

    def fake_run(model: Any, images: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        captured["n_in_multi"] = images.shape[0]
        return _fake_run(images.shape[0])

    def fake_single(
        model: Any, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1-view: returns (depth, point_map, K) per the helper's contract.
        captured["n_in_single"] = images.shape[0]
        h, w = 4, 5
        depth = np.full((1, h, w), 2.0, dtype=np.float32)
        pmap = np.zeros((1, h, w, 3), dtype=np.float32)
        K = np.array(
            [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32
        )[None]
        return depth, pmap, K

    monkeypatch.setattr(MonST3RAdapter, "_load", fake_load)
    monkeypatch.setattr(monst3r_mod, "_run_mast3r", fake_run)
    monkeypatch.setattr(monst3r_mod, "_monst3r_single_frame_eval", fake_single)


class TestConfig:
    def test_capabilities(self) -> None:
        a = MonST3RAdapter(device="cpu")
        assert a.capabilities.tasks == frozenset({"mono_depth", "mvs_depth", "pose"})
        assert a.capabilities.min_views == 1  # single-frame via duplication

    def test_config_hash_folds_checkpoint_and_ga(self) -> None:
        base = MonST3RAdapter(device="cpu").config_hash()
        other_ckpt = MonST3RAdapter(device="cpu", checkpoint="other/model").config_hash()
        other_ga = MonST3RAdapter(device="cpu", ga_niter=100).config_hash()
        assert base != other_ckpt
        assert base != other_ga


class TestPredict:
    def test_single_frame_uses_avg_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _patch(monkeypatch, captured)
        images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        pred = MonST3RAdapter(device="cpu").predict(images)
        # Adapter v1.1: single-frame routes through _monst3r_single_frame_eval
        # (MonST3R's eval_mono_depth shape), NOT the shared _run_mast3r path.
        assert "n_in_single" in captured and captured["n_in_single"] == 1
        assert "n_in_multi" not in captured  # PairViewer N=2 path skipped
        # Prediction is single-view: depth/extrinsics/point_map all (1, ...).
        assert pred.depth.shape[0] == 1
        assert pred.extrinsics.shape[0] == 1
        assert pred.point_map.shape[0] == 1
        # The new path doesn't emit confidence (PairViewer was the only source).
        assert pred.confidence is None
        assert pred.metadata["single_frame_duplicated"] is True
        assert pred.metadata["single_frame_path"] == "eval_mono_depth_avg"
        assert pred.metadata["flow_refinement"] is False

    def test_multi_view_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _patch(monkeypatch, captured)
        images = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        pred = MonST3RAdapter(device="cpu").predict(images)
        # Multi-view still routes through _run_mast3r unchanged.
        assert captured["n_in_multi"] == 3
        assert "n_in_single" not in captured
        assert pred.depth.shape[0] == 3
        assert pred.metadata["single_frame_duplicated"] is False
        assert pred.metadata["single_frame_path"] is None

    def test_max_views_cap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _patch(monkeypatch, captured)
        images = np.zeros((33, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="capped at"):
            MonST3RAdapter(device="cpu").predict(images)
