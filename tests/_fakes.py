"""Shared fake model + dataset implementations for tests.

Plain importable classes (no pytest magic). The fixtures that register these
in the global registries live in ``tests/conftest.py`` so they're auto-injected
without any per-file imports.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction

__all__ = [
    "_FakeDataset",
    "_FixedDepthModel",
    "_MultiViewPointCloudDataset",
    "_PointMapModel",
]


class _FixedDepthModel(Model):
    """Fake model that returns a constant AbsRel = ``target_abs_rel``."""

    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
    )

    def __init__(self, *, device: str = "cpu", target_abs_rel: float = 0.10) -> None:
        self.device = device
        # To produce abs_rel=t deterministically: pred = gt * (1 - t) for constant t.
        self.target_abs_rel = float(target_abs_rel)

    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> Prediction:
        n, h, w, _ = images.shape
        # gt will be 1.0 in our fake dataset; (1 - t) * 1.0 gives |1 - (1-t)|/1 = t.
        depth = np.full((n, h, w), 1.0 - self.target_abs_rel, dtype=np.float32)
        return Prediction(depth=depth)


class _PointMapModel(Model):
    """Fake multi-view model that returns an identity point map + extrinsics.

    ``predict`` builds a point map whose XYZ values reconstruct a trivial
    scene (depth=1 at every pixel for every view) in the first camera's
    frame, and returns extrinsics that place the cameras along +X at unit
    spacing. Lets reproduction tests exercise the chamfer / F-score path
    without needing GPU or HuggingFace downloads.
    """

    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth", "mvs_depth", "pose"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
    )

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> Prediction:
        n, h, w, _ = images.shape
        # Point map in the first camera's frame: every pixel at Z=1 m.
        pmap = np.zeros((n, h, w, 3), dtype=np.float32)
        pmap[..., 2] = 1.0
        ext = np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1))
        for i in range(n):
            ext[i, 0, 3] = float(i)  # cameras at x=0, 1, 2, ...
        return Prediction(point_map=pmap, extrinsics=ext)


class _FakeDataset(Dataset):
    split = "test"

    def __init__(self, *, n_samples: int = 3) -> None:
        self.n = n_samples

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(
                sample_id=f"s{i}",
                images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                intrinsics=np.eye(3, dtype=np.float32)[None],
                extrinsics_gt=np.eye(4, dtype=np.float32)[None],
                depth_gt=np.ones((1, 4, 4), dtype=np.float32),
            )

    def __len__(self) -> int:
        return self.n


class _MultiViewPointCloudDataset(Dataset):
    """3-view synthetic dataset with a GT point cloud — for chamfer tests.

    Cameras placed along +X at unit spacing (matching ``_PointMapModel``'s
    predicted extrinsic layout) so camera-centre Umeyama on pred → gt is
    the identity. GT point cloud is a tiny grid at Z=1; predicted point
    map has the same structure, so aligned chamfer is small.
    """

    split = "test"

    def __init__(self, *, n_samples: int = 2) -> None:
        self.n = n_samples

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            ext = np.tile(np.eye(4, dtype=np.float32)[None], (3, 1, 1))
            for v in range(3):
                ext[v, 0, 3] = float(v)
            pcd = np.array(
                [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]],
                dtype=np.float32,
            )
            yield Sample(
                sample_id=f"s{i}",
                images=np.zeros((3, 4, 4, 3), dtype=np.uint8),
                intrinsics=np.tile(np.eye(3, dtype=np.float32)[None], (3, 1, 1)),
                extrinsics_gt=ext,
                point_cloud_gt=pcd,
            )

    def __len__(self) -> int:
        return self.n
