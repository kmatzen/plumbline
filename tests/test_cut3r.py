"""Unit tests for the CUT3R adapter's plumbline-side conversion logic.

CUT3R itself can't run here (needs the repo + weights + GPU), but the parts
plumbline owns — preprocessing, SE(3) transform, and the output→conventions
mapping — are testable with synthetic data and a mocked backend. That mapping
is exactly where a coordinate / depth / pose-convention bug would hide, so it's
worth covering before the GPU validation run.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from plumbline.conventions import rebase_to_first_camera  # noqa: E402
from plumbline.models.cut3r import (  # noqa: E402
    CUT3RAdapter,
    _build_views,
    _transform_points,
)


class TestBuildViews:
    def test_shapes_keys_and_normalization(self) -> None:
        imgs = (np.random.default_rng(0).random((2, 480, 640, 3)) * 255).astype(np.uint8)
        views = _build_views(imgs, long_edge=512)
        assert len(views) == 2
        v = views[0]
        required = {
            "img",
            "ray_map",
            "true_shape",
            "idx",
            "instance",
            "camera_pose",
            "img_mask",
            "ray_mask",
            "update",
            "reset",
        }
        assert required <= set(v)

        img = v["img"]
        assert img.shape[0] == 1 and img.shape[1] == 3  # (1, 3, H, W)
        # dust3r 512-branch: long edge resized to 512, dims multiples of 16.
        assert max(img.shape[-2], img.shape[-1]) == 512
        assert img.shape[-1] % 16 == 0 and img.shape[-2] % 16 == 0
        # ImgNorm maps to [-1, 1].
        assert float(img.min()) >= -1.0001
        assert float(img.max()) <= 1.0001

    def test_view_extra_keys_match_demo(self) -> None:
        imgs = (np.random.default_rng(1).random((1, 240, 320, 3)) * 255).astype(np.uint8)
        v = _build_views(imgs, long_edge=512)[0]
        img = v["img"]
        # ray_map is all-NaN, (1, 6, H, W) per demo.prepare_input.
        assert v["ray_map"].shape == (1, 6, img.shape[-2], img.shape[-1])
        assert bool(torch.isnan(v["ray_map"]).all())
        # camera_pose seeded to identity (1, 4, 4).
        assert torch.allclose(v["camera_pose"][0], torch.eye(4))
        assert bool(v["img_mask"].item()) is True
        assert bool(v["ray_mask"].item()) is False
        assert bool(v["update"].item()) is True
        assert bool(v["reset"].item()) is False


class TestTransformPoints:
    def test_applies_se3_rotation_and_translation(self) -> None:
        pts = np.zeros((1, 2, 2, 3), dtype=np.float32)
        pts[0, 0, 0] = [1.0, 0.0, 5.0]  # a non-trivial point
        # 90° about +z, then translate.
        th = np.pi / 2
        E = np.eye(4, dtype=np.float64)[None]
        E[0, :3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
        E[0, :3, 3] = [10.0, 20.0, 30.0]
        out = _transform_points(E, pts)
        # R@(1,0,5) = (0,1,5); + t = (10, 21, 35)
        assert np.allclose(out[0, 0, 0], [10.0, 21.0, 35.0], atol=1e-5)
        # zero point maps to the translation
        assert np.allclose(out[0, 1, 1], [10.0, 20.0, 30.0], atol=1e-5)


class TestAdapterConfig:
    def test_size_224_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            CUT3RAdapter(device="cpu", size=224)

    def test_bad_size_raises(self) -> None:
        with pytest.raises(ValueError, match="size must be"):
            CUT3RAdapter(device="cpu", size=384)

    def test_capabilities(self) -> None:
        a = CUT3RAdapter(device="cpu")
        assert a.capabilities.tasks == frozenset({"mono_depth", "mvs_depth", "pose"})
        assert a.capabilities.min_views == 1
        assert a.capabilities.is_metric is True

    def test_config_hash_folds_checkpoint(self) -> None:
        h1 = CUT3RAdapter(device="cpu", checkpoint="/a/one.pth").config_hash()
        h2 = CUT3RAdapter(device="cpu", checkpoint="/a/two.pth").config_hash()
        assert h1 != h2

    def test_checkpoint_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUT3R_CKPT", "/custom/ckpt.pth")
        assert CUT3RAdapter(device="cpu").checkpoint == "/custom/ckpt.pth"


def _install_fake_cut3r_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    self_pts: np.ndarray,
    conf: np.ndarray,
    c2w: np.ndarray,
) -> None:
    """Inject fake ``src.dust3r.{model,inference,utils.camera}`` modules.

    ``inference`` returns one pred dict per view carrying the given self-view
    points, confidence, and a per-view pose *index* (decoded by the fake
    ``pose_encoding_to_camera`` into the supplied ``c2w``). Lets us drive
    ``CUT3RAdapter.predict`` end-to-end without the real model.
    """
    n = self_pts.shape[0]

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, _path: str) -> _FakeModel:
            return cls()

        def to(self, _device: str) -> _FakeModel:
            return self

        def eval(self) -> _FakeModel:
            return self

    def _inference(views: Any, model: Any, device: Any) -> tuple[dict[str, Any], Any]:
        preds = []
        for i in range(n):
            preds.append(
                {
                    "pts3d_in_self_view": torch.from_numpy(self_pts[i][None]).float(),
                    "conf_self": torch.from_numpy(conf[i][None]).float(),
                    # carry the view index in the "encoding"; the fake decoder
                    # uses it to look up the right c2w.
                    "camera_pose": torch.tensor([[float(i)]]),
                }
            )
        return {"pred": preds, "views": list(views)}, None

    def _pose_encoding_to_camera(enc: Any) -> Any:
        i = int(enc[0, 0].item())
        return torch.from_numpy(c2w[i][None]).float()

    model_mod = types.ModuleType("src.dust3r.model")
    model_mod.ARCroco3DStereo = _FakeModel  # type: ignore[attr-defined]
    inf_mod = types.ModuleType("src.dust3r.inference")
    inf_mod.inference = _inference  # type: ignore[attr-defined]
    cam_mod = types.ModuleType("src.dust3r.utils.camera")
    cam_mod.pose_encoding_to_camera = _pose_encoding_to_camera  # type: ignore[attr-defined]

    for name, mod in [
        ("src", types.ModuleType("src")),
        ("src.dust3r", types.ModuleType("src.dust3r")),
        ("src.dust3r.model", model_mod),
        ("src.dust3r.inference", inf_mod),
        ("src.dust3r.utils", types.ModuleType("src.dust3r.utils")),
        ("src.dust3r.utils.camera", cam_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


class TestPredictConversion:
    def test_predict_maps_outputs_to_conventions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rng = np.random.default_rng(7)
        # Two views, tiny maps. Positive z so depth is valid.
        h, w = 4, 5
        self_pts = rng.random((2, h, w, 3)).astype(np.float32)
        self_pts[..., 2] += 1.0  # ensure z > 0
        conf = rng.random((2, h, w)).astype(np.float32)
        # Distinct c2w per view (view 0 deliberately NOT identity, to prove rebase).
        c2w = np.tile(np.eye(4, dtype=np.float64)[None], (2, 1, 1))
        c2w[0, :3, 3] = [1.0, 2.0, 3.0]
        c2w[1, :3, 3] = [4.0, 6.0, 8.0]
        _install_fake_cut3r_backend(monkeypatch, self_pts=self_pts, conf=conf, c2w=c2w)

        images = (rng.random((2, 32, 40, 3)) * 255).astype(np.uint8)
        pred = CUT3RAdapter(device="cpu", checkpoint="/fake/ckpt.pth").predict(images)

        # depth = self-view z
        assert pred.depth.shape == (2, h, w)
        assert np.allclose(pred.depth, self_pts[..., 2], atol=1e-5)
        # confidence carried through
        assert pred.confidence is not None
        assert pred.confidence.shape == (2, h, w)

        # extrinsics rebased: view 0 is identity; view 1 = inv(c2w0) @ c2w1
        expected_E = rebase_to_first_camera(c2w)
        assert np.allclose(pred.extrinsics[0], np.eye(4), atol=1e-5)
        assert np.allclose(pred.extrinsics[1], expected_E[1], atol=1e-4)

        # point map consistent with (rebased E, self-view points)
        expected_pmap = _transform_points(expected_E, self_pts)
        assert pred.point_map.shape == (2, h, w, 3)
        assert np.allclose(pred.point_map, expected_pmap, atol=1e-4)

    def test_predict_masks_nonpositive_depth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        h, w = 3, 3
        self_pts = np.ones((1, h, w, 3), dtype=np.float32)
        self_pts[0, 0, 0, 2] = -5.0  # behind camera → invalid
        self_pts[0, 1, 1, 2] = np.nan  # non-finite → invalid
        conf = np.ones((1, h, w), dtype=np.float32)
        c2w = np.eye(4, dtype=np.float64)[None]
        _install_fake_cut3r_backend(monkeypatch, self_pts=self_pts, conf=conf, c2w=c2w)

        images = (np.zeros((1, 16, 16, 3))).astype(np.uint8)
        pred = CUT3RAdapter(device="cpu", checkpoint="/fake/ckpt.pth").predict(images)
        assert pred.depth[0, 0, 0] == 0.0
        assert pred.depth[0, 1, 1] == 0.0
        assert pred.depth[0, 2, 2] == 1.0
