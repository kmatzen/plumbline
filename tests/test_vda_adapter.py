"""VDA output-convention regression: relative variants must return DEPTH.

The relative Video-Depth-Anything checkpoints emit DISPARITY (inverse depth) from
the Depth-Anything-V2 DPT head. The harness's ``scale_shift`` alignment fits in
inverse-depth space (it does ``1/pred``), so the adapter must invert disparity to
depth before returning — otherwise a relative VDA cell silently fits the wrong
functional form (``s/D + b`` instead of MiDaS ``s*D + b``) and misrepresents the
model. The metric variants already emit meters and must pass through unchanged.
"""

from __future__ import annotations

import numpy as np

from plumbline.conventions import EPS
from plumbline.models.vda import VideoDepthAnythingAdapter


class _StubModel:
    """Stands in for the vendored VDA model; returns a fixed disparity map."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def infer_video_depth(self, frames, *args, **kwargs):
        return self._arr, 30.0


def _run(monkeypatch, variant: str, raw: np.ndarray) -> np.ndarray:
    adapter = VideoDepthAnythingAdapter(variant=variant, device="cpu")
    monkeypatch.setattr(adapter, "_load", lambda: None)
    adapter._model = _StubModel(raw)
    images = np.full((1, 4, 4, 3), 128, dtype=np.uint8)
    return adapter.predict(images).depth


def test_relative_variant_inverts_disparity_to_depth(monkeypatch) -> None:
    # Head output is disparity (larger = closer). Adapter must return its reciprocal.
    disp = np.array([[[1.0, 2.0, 4.0, 0.0]]], dtype=np.float32).reshape(1, 1, 4)
    depth = _run(monkeypatch, "vits", disp)
    expected = 1.0 / np.maximum(disp, EPS)
    np.testing.assert_allclose(depth, expected, rtol=1e-6)
    # Sanity: a far (small-disparity) pixel must map to a LARGER depth than a
    # near (large-disparity) pixel — the inversion is in the right direction.
    assert depth[0, 0, 0] > depth[0, 0, 2]


def test_metric_variant_passes_depth_through(monkeypatch) -> None:
    meters = np.array([[[0.5, 1.0, 3.0, 10.0]]], dtype=np.float32).reshape(1, 1, 4)
    depth = _run(monkeypatch, "metric-vits", meters)
    np.testing.assert_allclose(depth, meters, rtol=1e-6)
