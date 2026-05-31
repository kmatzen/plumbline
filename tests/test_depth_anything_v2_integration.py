"""Weights-gated integration tests for the DepthAnythingV2 adapter.

These tests require network access to HuggingFace Hub on first run
(weights are cached locally thereafter). They're skipped otherwise.
Run explicitly with::

    uv sync
    uv run pytest -m weights

Size: the ``small`` variant is ~25M parameters and fits on CPU. The test
runs DA-V2 on a tiny synthetic image and verifies:

- The HF pipeline loads weights without error.
- Output shapes, dtypes, and value ranges match the Prediction contract.
- Canonical-convention assertions pass on the output depth.
- End-to-end evaluation through ``runner.evaluate`` produces sane metrics
  when compared against synthetic ground truth.

This is the closest we can get to a real sanity check without renting a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="DA-V2 integration tests need the [models] extra")
pytest.importorskip("transformers", reason="DA-V2 integration tests need the [models] extra")

from plumbline.cache import PredictionCache
from plumbline.conventions import assert_valid_depth
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.depth_anything_v2 import DepthAnythingV2Adapter
from plumbline.runner import evaluate


def _dav2_paper_backend_available() -> bool:
    """True if the DA-V2 *paper* backend (the github repo on ``$DAV2_ROOT``)
    is importable.

    The tests below construct ``DepthAnythingV2Adapter`` with its default
    ``source="paper"``, which needs the cloned repo on ``sys.path``. Without
    it the only honest outcome is to skip (this module's docstring promises
    "skipped otherwise") — ``torch``/``transformers`` being present isn't
    sufficient. Guards against a hard failure on a fresh clone / dev laptop
    while still running on a configured GPU box.
    """
    try:
        from plumbline.models.depth_anything_v2 import _ensure_dav2_on_path

        _ensure_dav2_on_path()
        # Actually attempt the import the adapter performs — find_spec alone
        # only proves the package dir is on sys.path, not that it imports
        # (its dpt.py does `import cv2`, so a missing opencv-python makes the
        # adapter hard-fail). The module docstring promises a skip, not a
        # failure, when the backend isn't fully installed.
        import depth_anything_v2.dpt  # noqa: F401

        return True
    except Exception:
        return False


def _synthetic_image(h: int = 224, w: int = 224, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((1, h, w, 3)) * 255).astype(np.uint8)


class _SingleSampleDataset(Dataset):
    """One-sample dataset wrapping a numpy image and a GT depth map."""

    name = "synthetic-one"
    split = "test"

    def __init__(self, image: np.ndarray, depth_gt: np.ndarray) -> None:
        self._sample = Sample(
            sample_id="s0",
            images=image,
            intrinsics=np.eye(3, dtype=np.float32)[None],
            extrinsics_gt=np.eye(4, dtype=np.float32)[None],
            depth_gt=depth_gt,
        )

    def __iter__(self) -> Iterator[Sample]:
        yield self._sample

    def __len__(self) -> int:
        return 1


@pytest.mark.weights
@pytest.mark.skipif(
    not _dav2_paper_backend_available(),
    reason=(
        "DA-V2 paper backend not on this host — clone "
        "https://github.com/DepthAnything/Depth-Anything-V2 and set $DAV2_ROOT. "
        "These run on a configured GPU box; skipped on dev machines per the "
        "module docstring."
    ),
)
class TestDepthAnythingV2Real:
    def test_predict_cpu_synthetic(self) -> None:
        model = DepthAnythingV2Adapter(device="cpu", variant="small")
        img = _synthetic_image()
        pred = model.predict(img)

        assert pred.depth is not None
        assert pred.depth.shape == (1, 224, 224)
        assert pred.depth.dtype == np.float32
        assert_valid_depth(pred.depth, name="da-v2/cpu")
        assert pred.metadata["variant"] == "small"
        assert pred.metadata["alignment_hint"] == "scale_shift"

    def test_predict_mps_synthetic(self) -> None:
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this host")
        model = DepthAnythingV2Adapter(device="mps", variant="small")
        img = _synthetic_image()
        pred = model.predict(img)

        assert pred.depth.shape == (1, 224, 224)
        assert_valid_depth(pred.depth, name="da-v2/mps")

    def test_batched_predict(self) -> None:
        """Feed 3 images in one call and verify batched output."""
        model = DepthAnythingV2Adapter(device="cpu", variant="small")
        batch = np.concatenate([_synthetic_image(seed=i) for i in range(3)], axis=0)
        pred = model.predict(batch)
        assert pred.depth.shape == (3, 224, 224)
        # Each depth map should have its own distribution — not all-equal.
        assert not np.allclose(pred.depth[0], pred.depth[1])

    def test_end_to_end_through_runner(self, tmp_path: Path) -> None:
        """Run DA-V2 through evaluate() with synthetic GT + scale_shift alignment.

        On a random image the model output is meaningless, but the pipeline
        still exercises every seam: adapter → cache → metrics → report.
        """
        img = _synthetic_image(h=56, w=56)
        # Plausible GT: positive depth in meters, avoiding zeros.
        gt = np.full((1, 56, 56), 2.0, dtype=np.float32)

        dataset = _SingleSampleDataset(img, gt)
        model = DepthAnythingV2Adapter(device="cpu", variant="small")

        report = evaluate(
            model=model,
            dataset=dataset,
            tasks=["mono_depth"],
            scale_alignment="scale_shift",
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 1
        assert report.n_skipped == 0
        assert np.isfinite(report.aggregate_metrics["abs_rel"])
        assert np.isfinite(report.aggregate_metrics["rmse"])
        # Cache round-trip: a second run should not re-invoke the model.
        calls = {"n": 0}
        original = model.predict

        def counted(images, intrinsics=None):
            calls["n"] += 1
            return original(images, intrinsics)

        model.predict = counted  # type: ignore[method-assign]
        _ = evaluate(
            model=model,
            dataset=dataset,
            tasks=["mono_depth"],
            scale_alignment="median",
            cache=PredictionCache(tmp_path),
        )
        assert calls["n"] == 0
