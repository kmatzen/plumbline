"""Weights- and data-gated tests that run adapters against real photographs.

These bridge the gap between:
- synthetic-input weights tests (which we already have; they validate the
  model loader + forward pass but not that the adapter handles real
  photographic content)
- full paper-reproduction runs (which need GPU + auth-gated GT depth)

The data fixtures here are the Sintel final-pass images, which do **not**
require auth (unlike Sintel's depth + camera archives). If you have run::

    curl -fL -o /tmp/sintel.zip \\
      http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
    unzip /tmp/sintel.zip -d $SINTEL_ROOT

then these tests will pick up the alley_1 scene automatically. Otherwise
they skip cleanly.

What they validate (no GT depth needed):
- The adapter produces finite depth on real photographic content.
- Adjacent frames give temporally stable outputs — a broken
  preprocessing pipeline (wrong normalization, bad resize, channel swap)
  would show up as high frame-to-frame drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="integration tests need [models] extra")
pytest.importorskip("transformers", reason="integration tests need [models] extra")
from PIL import Image

from plumbline.conventions import assert_valid_depth
from plumbline.models.depth_anything_v2 import DepthAnythingV2Adapter


def _sintel_alley_1() -> Path | None:
    """Return the alley_1 final-pass directory if present, else None."""
    root = os.environ.get("SINTEL_ROOT")
    if not root:
        root = str(Path.home() / "data/sintel")
    scene = Path(root) / "training/final/alley_1"
    return scene if scene.exists() and any(scene.glob("frame_*.png")) else None


sintel_root = _sintel_alley_1()
pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(
        sintel_root is None,
        reason=(
            "Sintel alley_1 final-pass not found at $SINTEL_ROOT/training/final/alley_1. "
            "Download MPI-Sintel-complete.zip (optical-flow bundle, no auth) to enable."
        ),
    ),
]


def _load_real_frames(n: int) -> np.ndarray:
    assert sintel_root is not None
    frames = sorted(sintel_root.glob("frame_*.png"))[:n]
    return np.stack(
        [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in frames]
    )


class TestDepthAnythingV2RealImagery:
    def test_runs_on_real_photograph(self) -> None:
        """Single frame: inference completes and conventions hold on real content."""
        imgs = _load_real_frames(1)
        model = DepthAnythingV2Adapter(device="cpu", variant="small")
        pred = model.predict(imgs)
        assert pred.depth is not None
        assert pred.depth.shape == imgs.shape[:3]  # (N, H, W)
        assert_valid_depth(pred.depth)
        # Depth is derived from disparity; values must be finite and > 0.
        assert np.all(np.isfinite(pred.depth))
        assert float(pred.depth.min()) > 0

    def test_temporal_stability_across_adjacent_frames(self) -> None:
        """Nearby frames of a continuous shot should produce similar depth.

        A dead giveaway of a broken preprocessing pipeline (channel swap,
        normalization mismatch) is wildly diverging depth on near-identical
        inputs. The exact threshold is empirical; 20% of the total output
        range is very generous — real DA-V2 on Sintel alley_1 clocks in at
        <1% in our runs — but would fail loudly on a gross regression.
        """
        imgs = _load_real_frames(5)
        model = DepthAnythingV2Adapter(device="cpu", variant="small")
        pred = model.predict(imgs)

        flat = pred.depth.reshape(imgs.shape[0], -1)
        frame_diff = np.mean(np.abs(np.diff(flat, axis=0)), axis=1).mean()
        depth_range = float(pred.depth.max() - pred.depth.min())

        # Same-scene adjacent frames should diverge by <20% of the full range.
        assert frame_diff < depth_range * 0.20, (
            f"frame-to-frame depth diff {frame_diff:.4f} too large "
            f"relative to range {depth_range:.4f}; likely preprocessing bug"
        )
