"""iBims-1 indoor benchmark loader (via MoGe's preprocessed bundle).

iBims-1 (Koch et al. 2018, "Evaluation of CNN-based Single-Image
Depth Estimation Methods", arXiv:1805.01328) is a 100-image indoor
mono-depth benchmark: high-fidelity DSLR + laser-scanner ground truth
across rooms, corridors, restaurants, offices, lecture rooms, etc.
MoGe Table 1/2 reports iBims-1 as the canonical "high-quality indoor"
slot alongside NYU.

We load iBims-1 from MoGe's preprocessed HuggingFace bundle
``Ruicheng/monocular-geometry-evaluation`` (file ``iBims-1.zip``,
~40 MB, 100 scene subdirs). Format mirrors the GSO bundle exactly:

    <root>/<scene>/image.jpg          # 640x480 RGB
    <root>/<scene>/depth.png          # 640x480 uint16, log-encoded
                                      #   (per-image near/far in PNG info dict)
    <root>/<scene>/segmentation.png   # 640x480 uint8 semantic mask
    <root>/<scene>/meta.json          # {"intrinsics": [[fx/W, 0, cx/W], ...]}

Depth decoding uses the same ``read_moge_depth_png`` helper as GSO.
Intrinsics are normalized (fx/W, fy/H, cx/W, cy/H); the loader scales
them to pixel space.

Download (public, no auth)::

    pip install huggingface-hub
    hf download Ruicheng/monocular-geometry-evaluation \\
        --repo-type dataset --include 'iBims-1*' --local-dir data/moge_eval
    cd data/moge_eval && unzip iBims-1.zip
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    read_rgb_uint8,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.gso import read_moge_depth_png
from plumbline.datasets.registry import register_dataset

__all__ = ["IBims1Dataset"]


@register_dataset("ibims1")
class IBims1Dataset(Dataset):
    """iBims-1 indoor mono-depth loader (MoGe-bundle format).

    Each :class:`Sample` is a single 640x480 RGB frame with a dense
    GT depth map and per-image intrinsics. Extrinsics are identity
    (single-view).

    Parameters
    ----------
    root
        Directory containing ``<scene>/{image.jpg, depth.png,
        segmentation.png, meta.json}`` subdirs. Falls back to
        ``$IBIMS1_ROOT``.
    scenes
        Optional whitelist of scene names (e.g. ``["corridor_08",
        "lab_05"]``). ``None`` = all 100 scenes.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        scenes: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("IBIMS1_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "iBims-1 not found. Set --data-root or $IBIMS1_ROOT to a directory "
                "containing <scene>/{image.jpg, depth.png, meta.json}. Download "
                "from Ruicheng/monocular-geometry-evaluation on HuggingFace: "
                "`hf download Ruicheng/monocular-geometry-evaluation "
                "--repo-type dataset --include 'iBims-1*' --local-dir <root>/..` "
                "then `unzip iBims-1.zip`."
            )
        self.root = root_path
        all_scenes = sorted(
            p.name for p in root_path.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
        if scenes is not None:
            wanted = set(scenes)
            self.scene_names = [n for n in all_scenes if n in wanted]
        else:
            self.scene_names = all_scenes
        if not self.scene_names:
            raise DatasetNotAvailable(
                f"No iBims-1 scene subdirs found under {root_path}. Each subdir "
                "must contain meta.json. (Did you unzip iBims-1.zip into the "
                "right location?)"
            )

    def __iter__(self) -> Iterator[Sample]:
        for name in self.scene_names:
            yield self._load_sample(name)

    def __len__(self) -> int:
        return len(self.scene_names)

    def _load_sample(self, name: str) -> Sample:
        sample_dir = self.root / name
        img = read_rgb_uint8(sample_dir / "image.jpg")
        images = img[None]  # (1, H, W, 3)
        assert_valid_image(images, name=f"ibims1/{name}/image")

        H, W, _ = img.shape

        depth = read_moge_depth_png(sample_dir / "depth.png")
        if depth.shape != (H, W):
            raise ValueError(
                f"ibims1/{name}: depth {depth.shape} mismatches image {(H, W)}"
            )
        # MoGe-encoded depth uses NaN/inf for invalid/beyond-far. plumbline's
        # convention treats 0 as invalid; downstream metrics filter via
        # depth_valid AND-ed with depth>0.
        depth_valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(depth_valid, depth, 0.0).astype(np.float32)
        depth_gt = depth[None]

        with (sample_dir / "meta.json").open() as f:
            meta = json.load(f)
        K_norm = np.asarray(meta["intrinsics"], dtype=np.float64)
        # Normalised: fx/W, fy/H, cx/W, cy/H. Un-normalise to pixel K.
        K = K_norm.copy()
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H
        K_stack = K[None].astype(np.float32)
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"ibims1/{name}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"ibims1/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"ibims1/{name}/depth")

        # Optional segmentation mask — exposed via metadata so downstream
        # paper protocols (e.g. excluding sky/transparent classes) can pick
        # it up without the loader making protocol decisions itself.
        seg_path = sample_dir / "segmentation.png"
        seg: NDArray[np.uint8] | None = None
        if seg_path.exists():
            from PIL import Image as PImage

            seg_arr = np.asarray(PImage.open(seg_path), dtype=np.uint8)
            if seg_arr.shape == (H, W):
                seg = seg_arr

        return Sample(
            sample_id=f"ibims1/{name}",
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid[None],
            metadata={
                "scene": name,
                "split": self.split,
                "segmentation": seg,
            },
        )
