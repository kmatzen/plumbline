"""ETH3D mono-depth loader via MoGe's preprocessed bundle.

Distinct from the native ``eth3d`` loader (``datasets/eth3d.py``), which
serves multi-view high-res scenes for the VGGT/MASt3R **MVS-chamfer**
protocol. This loader serves ETH3D as **single-frame mono-depth** in the
exact form MoGe (Wang et al. 2024) evaluates for Table 3 — its
preprocessed HuggingFace bundle ``Ruicheng/monocular-geometry-evaluation``
(file ``ETH3D.zip``), same ``{image, depth.png, meta.json}`` layout as the
iBims-1 / GSO bundles:

    <root>/<scene>/image.{jpg,png}    # RGB (high-res; ETH3D frames are PNG)
    <root>/<scene>/depth.png          # uint16 log-encoded (MoGe format)
    <root>/<scene>/meta.json          # {"intrinsics": [[fx/W,0,cx/W], ...]}

Reads ``$ETH3D_MOGE_ROOT`` (NOT ``$ETH3D_ROOT``, the native chamfer set).
Decoding reuses ``read_moge_depth_png`` (shared with GSO / iBims-1).

NOTE (2026-05-28): created GPU-free to make the MoGe-Table-3 ETH3D cells
runnable; the loader logic mirrors the validated iBims-1 loader, but has
NOT yet been exercised against the real ETH3D.zip bundle — the first box
run validates it (image extension, resolution, intrinsics scaling).

Download (public, no auth)::

    hf download Ruicheng/monocular-geometry-evaluation \\
        --repo-type dataset --include 'ETH3D*' --local-dir data/moge_eval
    cd data/moge_eval && unzip ETH3D.zip      # → $ETH3D_MOGE_ROOT
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.datasets._common import DatasetNotAvailable, env_path, read_rgb_uint8
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.gso import read_moge_depth_png
from plumbline.datasets.registry import register_dataset

__all__ = ["ETH3DMogeEvalDataset"]


def _find_image(sample_dir: Path) -> Path | None:
    """MoGe bundles store RGB as image.jpg (iBims-1/GSO) or image.png
    (ETH3D's high-res frames). Accept either."""
    for ext in ("jpg", "png", "jpeg"):
        p = sample_dir / f"image.{ext}"
        if p.exists():
            return p
    return None


@register_dataset("eth3d-moge-eval")
class ETH3DMogeEvalDataset(Dataset):
    """ETH3D single-frame mono-depth from MoGe's eval bundle.

    Parameters
    ----------
    root
        Directory of ``<scene>/{image.*, depth.png, meta.json}`` subdirs.
        Falls back to ``$ETH3D_MOGE_ROOT``.
    scenes
        Optional whitelist of scene names. ``None`` = all scenes.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        scenes: list[str] | None = None,
        split: str = "test",
    ) -> None:
        # `split` is accepted for protocol-YAML compatibility with the other
        # MoGe-bundle loaders; the ETH3D MoGe eval bundle is a single set.
        if split != "test":
            raise ValueError(f"ETH3DMogeEvalDataset only exposes the test split; got {split!r}")
        root_path = Path(root) if root else env_path("ETH3D_MOGE_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ETH3D (MoGe bundle) not found. Set --data-root or "
                "$ETH3D_MOGE_ROOT to the unzipped ETH3D.zip from "
                "Ruicheng/monocular-geometry-evaluation (`hf download "
                "Ruicheng/monocular-geometry-evaluation --repo-type dataset "
                "--include 'ETH3D*' --local-dir <root>/..`). This is the MoGe "
                "mono-depth bundle, distinct from $ETH3D_ROOT (native chamfer)."
            )
        self.root = root_path
        all_scenes = sorted(
            p.name for p in root_path.iterdir() if p.is_dir() and (p / "meta.json").exists()
        )
        self.scene_names = (
            [n for n in all_scenes if n in set(scenes)] if scenes is not None else all_scenes
        )
        if not self.scene_names:
            raise DatasetNotAvailable(
                f"No ETH3D MoGe-bundle scene subdirs (with meta.json) under "
                f"{root_path}. Did you unzip ETH3D.zip into the right location?"
            )

    def __iter__(self) -> Iterator[Sample]:
        for name in self.scene_names:
            yield self._load_sample(name)

    def __len__(self) -> int:
        return len(self.scene_names)

    def _load_sample(self, name: str) -> Sample:
        sample_dir = self.root / name
        img_path = _find_image(sample_dir)
        if img_path is None:
            raise ValueError(f"eth3d-moge/{name}: no image.{{jpg,png}} in {sample_dir}")
        img = read_rgb_uint8(img_path)
        images = img[None]
        assert_valid_image(images, name=f"eth3d-moge/{name}/image")
        H, W, _ = img.shape

        depth = read_moge_depth_png(sample_dir / "depth.png")
        if depth.shape != (H, W):
            raise ValueError(f"eth3d-moge/{name}: depth {depth.shape} mismatches image {(H, W)}")
        depth_valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(depth_valid, depth, 0.0).astype(np.float32)
        depth_gt = depth[None]

        with (sample_dir / "meta.json").open() as f:
            meta = json.load(f)
        K = np.asarray(meta["intrinsics"], dtype=np.float64).copy()  # normalized
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H
        K_stack = K[None].astype(np.float32)
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"eth3d-moge/{name}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"eth3d-moge/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"eth3d-moge/{name}/depth")

        return Sample(
            sample_id=f"eth3d-moge/{name}",
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid[None],
            metadata={"scene": name, "split": self.split},
        )
