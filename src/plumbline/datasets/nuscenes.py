"""nuScenes validation loader for Depth Pro Table 1 metric δ₁.

Appendix Table 16: 881 random val samples, valid depth 0.001–80 m, GT
resolution 900×1600 (CAM_FRONT native). GT is sparse LiDAR projected to the
image plane (single ``LIDAR_TOP`` sweep per keyframe).

Requires ``nuscenes-devkit`` (``uv pip install nuscenes-devkit pyquaternion``)
and a local nuScenes tree at ``$NUSCENES_ROOT`` (v1.0-trainval for the paper
subset; v1.0-mini works for smoke tests with ``subset_size`` overridden).

The paper draws 881 frames from ~35K val camera keyframes; this loader uses
**CAM_FRONT** keyframes only (~6K val frames) with ``numpy`` RNG seed **42**,
which is a documented protocol approximation until Apple publishes the exact
index list.

Download::

    ./scripts/download-nuscenes.sh mini          # ~4 GB smoke
    ./scripts/download-nuscenes.sh depth-pro-val # instructions + metadata
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.datasets._common import DatasetNotAvailable, env_path, read_rgb_uint8
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["NuscenesDataset", "project_nuscenes_lidar_depth"]

_DEFAULT_SUBSET_SIZE = 881
_DEFAULT_SUBSET_SEED = 42
_DEFAULT_CAMERA = "CAM_FRONT"
_DEFAULT_VERSION = "v1.0-trainval"


def _require_nuscenes() -> tuple[Any, Any]:
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise DatasetNotAvailable(
            "nuscenes-devkit is required. Install with: uv pip install nuscenes-devkit pyquaternion"
        ) from exc
    return NuScenes, create_splits_scenes


def _val_cam_front_sample_tokens(
    nusc: object,
    *,
    version: str,
    camera_channel: str,
) -> list[str]:
    """Sorted val-split CAM_FRONT keyframe sample tokens."""
    _, create_splits_scenes = _require_nuscenes()
    if "mini" in version:
        val_scenes = set(create_splits_scenes()["mini_val"])
    else:
        val_scenes = set(create_splits_scenes()["val"])

    tokens: list[str] = []
    for sample in nusc.sample:  # type: ignore[attr-defined]
        scene = nusc.get("scene", sample["scene_token"])  # type: ignore[attr-defined]
        if scene["name"] not in val_scenes:
            continue
        cam_sd = nusc.get("sample_data", sample["data"][camera_channel])  # type: ignore[attr-defined]
        if not cam_sd["is_key_frame"]:
            continue
        tokens.append(sample["token"])
    tokens.sort()
    return tokens


def _subset_manifest_path(root: Path, *, version: str, camera: str, seed: int, n: int) -> Path:
    tag = version.replace(".", "")
    return root / ".plumbline_manifest" / f"nuscenes_{tag}_{camera}_seed{seed}_n{n}.json"


def project_nuscenes_lidar_depth(
    nusc: object,
    sample_token: str,
    *,
    camera_channel: str = _DEFAULT_CAMERA,
    min_dist_m: float = 0.1,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32], tuple[int, int]]:
    """Project ``LIDAR_TOP`` onto ``camera_channel``; return depth (m), valid, K, (h, w)."""
    from nuscenes.nuscenes import NuScenesExplorer

    sample = nusc.get("sample", sample_token)  # type: ignore[attr-defined]
    cam_token = sample["data"][camera_channel]
    lidar_token = sample["data"]["LIDAR_TOP"]
    cam_sd = nusc.get("sample_data", cam_token)  # type: ignore[attr-defined]
    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])  # type: ignore[attr-defined]
    K = np.array(cs["camera_intrinsic"], dtype=np.float32)

    explorer = NuScenesExplorer(nusc)
    points2d, depths, image = explorer.map_pointcloud_to_image(
        lidar_token,
        cam_token,
        min_dist=min_dist_m,
    )
    w, h = image.size
    depth = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.bool_)

    xs = np.round(points2d[0]).astype(np.int32)
    ys = np.round(points2d[1]).astype(np.int32)
    ds = np.asarray(depths, dtype=np.float32)
    for x, y, d in zip(xs, ys, ds, strict=True):
        if d <= 0 or not np.isfinite(d):
            continue
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if (not valid[y, x]) or d < depth[y, x]:
            depth[y, x] = d
            valid[y, x] = True

    return depth, valid, K, (h, w)


@register_dataset("nuscenes")
class NuscenesDataset(Dataset):
    """nuScenes val CAM_FRONT metric depth (Depth Pro Table 16 lineage).

    Parameters
    ----------
    root
        nuScenes dataroot (``samples/``, ``v1.0-*``). Defaults to ``$NUSCENES_ROOT``.
    version
        ``v1.0-trainval`` (paper) or ``v1.0-mini`` (smoke).
    split
        Only ``val`` (and ``mini_val`` when ``version`` contains ``mini``).
    camera_channel
        Camera name (default ``CAM_FRONT``).
    subset_size
        Random subset count (default 881).
    subset_seed
        RNG seed for subset selection (default 42).
    max_depth_invalid
        Pixels above this depth (m) are marked invalid.
    cache_depth
        Cache projected depth under ``<root>/.plumbline_manifest/nuscenes_depth/``.
    """

    split: str = "val"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        version: str = _DEFAULT_VERSION,
        split: str = "val",
        camera_channel: str = _DEFAULT_CAMERA,
        subset_size: int = _DEFAULT_SUBSET_SIZE,
        subset_seed: int = _DEFAULT_SUBSET_SEED,
        max_depth_invalid: float = 80.0,
        cache_depth: bool = True,
    ) -> None:
        if split not in ("val", "mini_val"):
            raise ValueError(f"NuscenesDataset only supports val; got {split!r}")

        root_path = Path(root) if root else env_path("NUSCENES_ROOT")
        if root_path is None or not root_path.is_dir():
            raise DatasetNotAvailable(
                "nuScenes not found. Set --data-root or $NUSCENES_ROOT. "
                "Run ./scripts/download-nuscenes.sh"
            )

        meta = root_path / version
        if not meta.is_dir():
            raise DatasetNotAvailable(
                f"Missing nuScenes metadata folder {meta}. Run ./scripts/download-nuscenes.sh"
            )

        NuScenes, _ = _require_nuscenes()
        try:
            self.nusc = NuScenes(version=version, dataroot=str(root_path), verbose=False)
        except Exception as exc:
            raise DatasetNotAvailable(f"Failed to open nuScenes at {root_path}: {exc}") from exc

        manifest = _subset_manifest_path(
            root_path,
            version=version,
            camera=camera_channel,
            seed=subset_seed,
            n=subset_size,
        )
        if manifest.is_file():
            self.sample_tokens = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            pool = _val_cam_front_sample_tokens(
                self.nusc, version=version, camera_channel=camera_channel
            )
            if len(pool) < subset_size:
                raise DatasetNotAvailable(
                    f"Only {len(pool)} val {camera_channel} frames in {version}; "
                    f"need {subset_size}. Use v1.0-trainval or lower subset_size."
                )
            rng = np.random.default_rng(subset_seed)
            pick = rng.choice(len(pool), size=subset_size, replace=False)
            self.sample_tokens = [pool[int(i)] for i in sorted(pick)]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(self.sample_tokens, indent=0) + "\n",
                encoding="utf-8",
            )

        self.root = root_path
        self.version = version
        self.camera_channel = camera_channel
        self.max_depth_invalid = max_depth_invalid
        self.cache_depth = cache_depth
        self._depth_cache_dir = root_path / ".plumbline_manifest" / "nuscenes_depth"

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __iter__(self) -> Iterator[Sample]:
        for token in self.sample_tokens:
            yield self._load_sample(token)

    def _load_sample(self, sample_token: str) -> Sample:
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][self.camera_channel]
        cam_sd = self.nusc.get("sample_data", cam_token)
        img_path = self.root / cam_sd["filename"]
        name = Path(cam_sd["filename"]).stem

        cache_path = self._depth_cache_dir / f"{sample_token}_{self.camera_channel}.npz"
        if self.cache_depth and cache_path.is_file():
            cached = np.load(cache_path)
            depth = cached["depth"].astype(np.float32)
            valid = cached["valid"].astype(np.bool_)
            K = cached["intrinsics"].astype(np.float32)
            h, w = int(cached["height"]), int(cached["width"])
        else:
            depth, valid, K, (h, w) = project_nuscenes_lidar_depth(
                self.nusc,
                sample_token,
                camera_channel=self.camera_channel,
            )
            if self.cache_depth:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    depth=depth,
                    valid=valid,
                    intrinsics=K,
                    height=h,
                    width=w,
                )

        img = read_rgb_uint8(img_path)
        if img.shape[0] != h or img.shape[1] != w:
            raise ValueError(f"nuscenes/{name}: image {img.shape[:2]} != lidar projection {(h, w)}")

        images = img[None]
        assert_valid_image(images, name=f"nuscenes/{name}")

        valid &= np.isfinite(depth) & (depth > 0) & (depth < self.max_depth_invalid)
        depth_gt = np.where(valid, depth, 0.0).astype(np.float32)[None]
        depth_valid = valid[None]

        e_eye = np.eye(4, dtype=np.float32)[None]
        assert_valid_intrinsics(K[None], name=f"nuscenes/{name}/intrinsics")
        assert_valid_extrinsics(e_eye, name=f"nuscenes/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"nuscenes/{name}/depth")

        return Sample(
            sample_id=f"nuscenes/{name}",
            images=images,
            intrinsics=K[None],
            extrinsics_gt=e_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "sample_token": sample_token,
                "camera": self.camera_channel,
                "version": self.version,
                "image_size": (h, w),
            },
        )
