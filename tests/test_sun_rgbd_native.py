"""Tests for the native SUN RGB-D loader + Depth Pro GT-focal opt-in."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.datasets.sun_rgbd_native import read_sun_rgbd_native_depth
from plumbline.models.depth_pro import DepthProAdapter


def test_registered() -> None:
    assert "sun-rgbd-native" in DATASET_REGISTRY


def test_bitrotation_decode(tmp_path) -> None:
    # Native SUN RGB-D encodes depth as (mm << 3); recover via (raw>>3)|(raw<<13)
    # then /1000. Low 3 bits are zero, so the rotation is a pure >>3 here.
    # (mm << 3 must fit in uint16, so the encodable max is ~8.19 m — real data
    # is kinect-range, ≤8 m.) raw 65000 decodes to 8.125 m and clips to 8.
    raw = np.array([[8000, 16000], [64000, 65000]], dtype=np.uint16)
    p = tmp_path / "d.png"
    Image.fromarray(raw).save(p)
    d = read_sun_rgbd_native_depth(p)
    assert np.allclose(d, np.array([[1.0, 2.0], [8.0, 8.0]], dtype=np.float32))


def _make_root(tmp_path, n=3):
    root = tmp_path / "native"
    for sub in ("rgb", "depth", "intrinsics"):
        (root / sub).mkdir(parents=True)
    for i in range(1, n + 1):
        sid = f"img-{i:06d}"
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(root / "rgb" / f"{sid}.jpg")
        Image.fromarray((np.full((8, 10), 1500, dtype=np.uint16)) << 3).save(
            root / "depth" / f"{sid}.png"
        )
        (root / "intrinsics" / f"{sid}.txt").write_text(
            "500.0 0.0 5.0\n0.0 500.0 4.0\n0.0 0.0 1.0\n"
        )
    return root


def test_loader_yields_gt_focal(tmp_path) -> None:
    root = _make_root(tmp_path)
    ds = DATASET_REGISTRY["sun-rgbd-native"](root=root)
    assert len(ds) == 3
    s = next(iter(ds))
    assert s.images.shape == (1, 8, 10, 3)
    assert s.intrinsics[0, 0, 0] == pytest.approx(500.0)  # GT fx from intrinsics.txt
    assert np.allclose(s.depth_gt[0][s.depth_valid[0]], 1.5)


def test_adapter_gt_focal_opt_in() -> None:
    default = DepthProAdapter()
    gt = DepthProAdapter(use_gt_focal=True)
    assert default.capabilities.requires_intrinsics is False
    assert gt.capabilities.requires_intrinsics is True
    # default class-level capabilities must be untouched (no Booster regression).
    assert DepthProAdapter.capabilities.requires_intrinsics is False
    assert default.config_hash() != gt.config_hash()
