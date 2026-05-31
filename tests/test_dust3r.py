"""Unit tests for the DUSt3R adapter's plumbline-side routing.

DUSt3R delegates the actual dust3r global alignment to the shared
``_run_mast3r`` runner (covered by the MASt3R tests). What's adapter-
specific and worth testing here without a GPU/repo: the v1.1 single-
frame ``F(I, I)`` branch (DUSt3R paper §4.3 monocular protocol),
multi-view pass-through, and capability/cap handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plumbline.models import dust3r as dust3r_mod  # noqa: E402
from plumbline.models.dust3r import DUSt3RAdapter  # noqa: E402


def _patch(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Stub _load (no real model), the multi-view runner, and the new
    single-frame helper. Records what each was called with."""

    def fake_load(self: DUSt3RAdapter) -> None:
        self._model = object()

    def fake_run(model: Any, images: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        captured["n_in_multi"] = images.shape[0]
        n, h, w = images.shape[0], 4, 5
        return {
            "depth": np.full((n, h, w), 2.0, dtype=np.float32),
            "intrinsics": np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1)),
            "extrinsics": np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1)),
            "point_map": np.zeros((n, h, w, 3), dtype=np.float32),
            "confidence": np.ones((n, h, w), dtype=np.float32),
        }

    def fake_single(
        model: Any, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captured["n_in_single"] = images.shape[0]
        h, w = 4, 5
        depth = np.full((1, h, w), 2.0, dtype=np.float32)
        pmap = np.zeros((1, h, w, 3), dtype=np.float32)
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)[None]
        return depth, pmap, K

    monkeypatch.setattr(DUSt3RAdapter, "_load", fake_load)
    monkeypatch.setattr(dust3r_mod, "_run_mast3r", fake_run)
    monkeypatch.setattr(dust3r_mod, "_dust3r_single_frame_eval", fake_single)


class TestSingleFrame:
    def test_single_frame_uses_view_duplicate_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DUSt3R v1.1 + paper §4.3: N=1 must route through
        ``_dust3r_single_frame_eval`` (the F(I, I) view-duplicate trick),
        NOT raise as it did in v1.0.
        """
        captured: dict[str, Any] = {}
        _patch(monkeypatch, captured)
        images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        pred = DUSt3RAdapter(device="cpu").predict(images)
        assert "n_in_single" in captured and captured["n_in_single"] == 1
        assert "n_in_multi" not in captured  # multi-view path skipped
        assert pred.depth.shape[0] == 1
        assert pred.extrinsics.shape[0] == 1
        assert pred.point_map.shape[0] == 1
        # The single-frame path doesn't synthesise a confidence map.
        assert pred.confidence is None
        assert pred.metadata["single_frame_duplicated"] is True
        assert pred.metadata["single_frame_path"] == "eval_mono_depth_avg"
        assert pred.metadata["n_views"] == 1

    def test_multi_view_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """N>=2 must continue to route through the shared ``_run_mast3r``
        runner (CO3Dv2 pose path, unchanged from v1.0)."""
        captured: dict[str, Any] = {}
        _patch(monkeypatch, captured)
        images = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        pred = DUSt3RAdapter(device="cpu").predict(images)
        assert captured["n_in_multi"] == 3
        assert "n_in_single" not in captured
        assert pred.depth.shape[0] == 3
        assert pred.metadata["single_frame_duplicated"] is False
        assert pred.metadata["single_frame_path"] is None
        assert pred.metadata["n_views"] == 3
