"""End-to-end test of the multi-view-stereo pipeline with a fake adapter.

Before we rent a GPU to wire a real MVS model (VGGT, MASt3R, DA3), it's
worth exercising every code path the runner hits for multi-view work:

- Multi-view sample iteration + view-count capping
- Pose conversion / rebasing to first camera
- Pose metric computation (rotation, translation-cos, AUC)
- Point-cloud metric computation (Chamfer, F-score)

A synthetic adapter + synthetic dataset here drives the runner through
those seams without needing any upstream package. If this test passes,
the runner's MVS path is known-good; a failure on a real adapter must be
in the adapter, not in plumbline's pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from plumbline.cache import PredictionCache
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.runner import evaluate


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


def _rebase_to_first(poses: np.ndarray) -> np.ndarray:
    """world_from_camera so poses[0] is identity."""
    from plumbline.conventions import invert_pose

    inv0 = invert_pose(poses[0])
    return (inv0[None, ...] @ poses).astype(np.float32)


class _FakeMVSDataset(Dataset):
    """3 samples, 4 views each, with randomized but valid poses and a
    point cloud GT placed at known 3D positions.

    Depth is constant (1.5 m everywhere) so the runner exercises depth
    metrics without numerical surprises.
    """

    name = "fake-mvs"
    split = "test"

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._samples: list[Sample] = []
        for s_idx in range(3):
            # 4 views with random world-from-camera transforms, then rebase
            # so view 0 is identity.
            poses = np.stack(
                [_pose(_random_rotation(s_idx * 10 + i), rng.uniform(-1, 1, size=3))
                 for i in range(4)]
            )
            extrinsics = _rebase_to_first(poses)
            intrinsics = np.tile(
                np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
                (4, 1, 1),
            )
            depth = np.full((4, 16, 16), 1.5, dtype=np.float32)
            pcd = rng.uniform(-2, 2, size=(200, 3)).astype(np.float32)
            self._samples.append(
                Sample(
                    sample_id=f"scene_{s_idx}",
                    images=np.zeros((4, 16, 16, 3), dtype=np.uint8),
                    intrinsics=intrinsics,
                    extrinsics_gt=extrinsics,
                    depth_gt=depth,
                    point_cloud_gt=pcd,
                )
            )

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class _FakeMVSAdapter(Model):
    """Returns canonical shapes for every prediction field.

    Poses are identity (so rotation/translation errors are finite and
    non-trivial); depth is constant 1.5 m; the point map is zero-filled
    (mapped to the world origin). These are deliberately "wrong" outputs
    — they let us assert metrics are *computed* (no NaN propagation, no
    crashes, all-finite aggregates) without hand-computing expected
    numbers.
    """

    name = "fake-mvs"
    version = "test"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mvs_depth", "pose"}),
        is_metric=True,
        min_views=2,
        max_views=math.inf,
        requires_intrinsics=False,
    )

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> Prediction:
        n, h, w, _ = images.shape
        return Prediction(
            depth=np.full((n, h, w), 1.5, dtype=np.float32),
            intrinsics=(
                intrinsics.astype(np.float32)
                if intrinsics is not None
                else np.tile(
                    np.array([[500, 0, w / 2], [0, 500, h / 2], [0, 0, 1]], dtype=np.float32),
                    (n, 1, 1),
                )
            ),
            extrinsics=np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1)),
            point_map=np.zeros((n, h, w, 3), dtype=np.float32),
        )


class TestMVSPipeline:
    def test_pose_metrics_computed(self, tmp_path: Path) -> None:
        report = evaluate(
            model=_FakeMVSAdapter(),
            dataset=_FakeMVSDataset(),
            tasks=["pose"],
            cache=PredictionCache(tmp_path),
        )
        assert report.n_evaluated == 3
        assert report.n_skipped == 0
        for key in ("rotation_error_deg_mean", "translation_cos_err_deg_mean"):
            assert np.isfinite(report.aggregate_metrics[key])
        for t in (5.0, 10.0, 30.0):
            assert 0.0 <= report.aggregate_metrics[f"pose_auc@{t:g}"] <= 1.0

    def test_mvs_depth_metrics_computed(self, tmp_path: Path) -> None:
        report = evaluate(
            model=_FakeMVSAdapter(),
            dataset=_FakeMVSDataset(),
            tasks=["mvs_depth"],
            scale_alignment="none",
            cache=PredictionCache(tmp_path),
        )
        # Depth pred == GT (1.5 everywhere) → AbsRel should be 0.
        assert report.aggregate_metrics["abs_rel"] == 0.0

    def test_point_cloud_metrics_computed(self, tmp_path: Path) -> None:
        report = evaluate(
            model=_FakeMVSAdapter(),
            dataset=_FakeMVSDataset(),
            tasks=["mvs_depth"],
            f_score_threshold=1.0,
            cache=PredictionCache(tmp_path),
        )
        # Chamfer + F-score must appear whenever point_map + point_cloud_gt present.
        assert "chamfer" in report.aggregate_metrics
        assert "precision" in report.aggregate_metrics
        assert "recall" in report.aggregate_metrics
        assert "f_score" in report.aggregate_metrics
        # All-zero prediction vs random GT → precision is meaningful;
        # chamfer is finite and positive.
        assert np.isfinite(report.aggregate_metrics["chamfer"])
        assert report.aggregate_metrics["chamfer"] > 0

    def test_pose_metrics_skip_when_extrinsics_missing(self, tmp_path: Path) -> None:
        class _NoPoseAdapter(_FakeMVSAdapter):
            name = "fake-nopose"

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                p = super().predict(images, intrinsics)
                p.extrinsics = None
                return p

        report = evaluate(
            model=_NoPoseAdapter(),
            dataset=_FakeMVSDataset(),
            tasks=["pose", "mvs_depth"],
            scale_alignment="none",
            cache=PredictionCache(tmp_path),
        )
        # No extrinsics → no pose metrics, but depth still works.
        assert "rotation_error_deg_mean" not in report.aggregate_metrics
        assert report.aggregate_metrics["abs_rel"] == 0.0

    def test_point_cloud_metrics_skip_when_point_map_missing(
        self, tmp_path: Path
    ) -> None:
        class _NoPointMapAdapter(_FakeMVSAdapter):
            name = "fake-nopointmap"

            def predict(self, images: np.ndarray, intrinsics: np.ndarray | None = None):
                p = super().predict(images, intrinsics)
                p.point_map = None
                return p

        report = evaluate(
            model=_NoPointMapAdapter(),
            dataset=_FakeMVSDataset(),
            tasks=["mvs_depth"],
            scale_alignment="none",
            cache=PredictionCache(tmp_path),
        )
        assert "chamfer" not in report.aggregate_metrics
        assert "abs_rel" in report.aggregate_metrics

    def test_view_count_is_capped_at_max_views(self, tmp_path: Path) -> None:
        """Runner's ``max_views`` trims samples before the adapter sees them."""
        captured: list[int] = []
        base = _FakeMVSAdapter()

        def capturing_predict(images: np.ndarray, intrinsics: np.ndarray | None = None):
            captured.append(images.shape[0])
            return base.predict(images, intrinsics)

        base.predict = capturing_predict  # type: ignore[method-assign]

        evaluate(
            model=base,
            dataset=_FakeMVSDataset(),
            tasks=["pose"],
            max_views=2,
            cache=PredictionCache(tmp_path),
        )
        assert all(n == 2 for n in captured), f"expected all calls with 2 views, saw {captured}"
