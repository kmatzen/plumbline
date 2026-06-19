"""ETH3D mono-depth loader via MoGe's preprocessed bundle.

Distinct from the native ``eth3d`` loader (``datasets/eth3d.py``), which
serves multi-view high-res scenes for the VGGT/MASt3R **MVS-chamfer**
protocol. This loader serves ETH3D as **single-frame mono-depth** in the
exact form MoGe (Wang et al. 2024) evaluates for Table 3 — its
preprocessed HuggingFace bundle ``Ruicheng/monocular-geometry-evaluation``
(file ``ETH3D.zip``).

Like the DIODE / KITTI MoGe-eval loaders (and unlike GSO / iBims-1, whose
near-identity FoV makes a naive read sufficient), this loader **delegates
to MoGe's own** ``moge.test.dataloader.EvalDataLoaderPipeline._process_instance``
so the homographic FoV-warp is applied bit-identically to the paper-eval
pipeline. MoGe's ``configs/eval/all_benchmarks.json`` ETH3D entry warps the
high-res source frames to ``(width=2048, height=1365)`` (no ``drop_max_depth``
override → MoGe's 1000 m default).
https://github.com/microsoft/MoGe/blob/main/configs/eval/all_benchmarks.json

On-disk layout (mirrors the DIODE bundle — a nested, ``.index.txt``-driven
tree, NOT the flat ``<root>/<scene>`` of GSO / iBims-1)::

    <root>/ETH3D/.index.txt              # 453 sample subpaths, one per line
    <root>/ETH3D/<scene>/<frame>/image.jpg
    <root>/ETH3D/<scene>/<frame>/depth.png    # uint16 log-encoded (MoGe format)
    <root>/ETH3D/<scene>/<frame>/meta.json    # {"intrinsics": [[fx/W,0,cx/W], ...]}

Reads ``$ETH3D_MOGE_ROOT`` (the directory *containing* ``ETH3D/``; NOT
``$ETH3D_ROOT``, the native chamfer set).

Download (public, no auth)::

    hf download Ruicheng/monocular-geometry-evaluation \\
        --repo-type dataset --include 'ETH3D*' --local-dir data/moge_eval
    cd data/moge_eval && unzip ETH3D.zip      # → $ETH3D_MOGE_ROOT
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
)
from plumbline.datasets._common import DatasetNotAvailable, env_path
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["ETH3DMogeEvalDataset"]


def _find_image(sample_dir: Path) -> Path | None:
    """MoGe bundles store RGB as image.jpg (iBims-1/GSO/ETH3D) or image.png.
    Accept either."""
    for ext in ("jpg", "png", "jpeg"):
        p = sample_dir / f"image.{ext}"
        if p.exists():
            return p
    return None


@register_dataset("eth3d-moge-eval")
class ETH3DMogeEvalDataset(Dataset):
    """ETH3D single-frame mono-depth from MoGe's eval bundle.

    Delegates the homographic FoV-warp to MoGe's own eval pipeline so the
    eval set + preprocessing match MoGe Table 3 bit-for-bit.

    Parameters
    ----------
    root
        Directory *containing* the ``ETH3D/`` subtree (``ETH3D/.index.txt``
        plus the ``<scene>/<frame>/{image,depth,meta}`` dirs). Falls back to
        ``$ETH3D_MOGE_ROOT``.
    scenes
        Optional whitelist of scene names (the first ``.index.txt`` path
        component, e.g. ``["courtyard", "facade"]``). ``None`` = all scenes.
    split
        Accepted for protocol-YAML compatibility; the bundle is a single
        eval set, so only ``"test"`` is meaningful.
    """

    # MoGe's configs/eval/all_benchmarks.json ETH3D target FoV-warp.
    TARGET_WIDTH: int = 2048
    TARGET_HEIGHT: int = 1365
    DROP_MAX_DEPTH: float = 1000.0

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
        if root_path is None or not (root_path / "ETH3D").exists():
            raise DatasetNotAvailable(
                "ETH3D (MoGe bundle) not found. Set --data-root or "
                "$ETH3D_MOGE_ROOT to a directory containing ETH3D/.index.txt "
                "plus the unzipped ETH3D/<scene>/<frame>/... tree. Stage via: "
                "hf download Ruicheng/monocular-geometry-evaluation "
                "--repo-type dataset --include 'ETH3D*' --local-dir <tmp> && "
                "unzip <tmp>/ETH3D.zip -d $ETH3D_MOGE_ROOT. This is the MoGe "
                "mono-depth bundle, distinct from $ETH3D_ROOT (native chamfer)."
            )
        self.root = root_path

        # Build a MoGe EvalDataLoaderPipeline handle for its
        # ``_process_instance`` warp logic only (we don't ``.start()`` the
        # worker processes). Mirrors DIODEMogeEvalLoader / KITTIMogeEvalLoader.
        try:
            from plumbline.models.moge import _ensure_moge_on_path

            _ensure_moge_on_path()  # vendored moge + its pinned utils3d/pipeline
            from moge.test.dataloader import EvalDataLoaderPipeline
        except ModuleNotFoundError as exc:
            raise ImportError(
                "ETH3DMogeEvalDataset needs the `moge` package for its "
                "homographic warp (matches the paper's eval pipeline). "
                "Install with `uv pip install "
                "'git+https://github.com/microsoft/MoGe.git'`."
            ) from exc
        self._moge_pipe = EvalDataLoaderPipeline(
            path=str(root_path / "ETH3D"),
            width=self.TARGET_WIDTH,
            height=self.TARGET_HEIGHT,
            split=".index.txt",
            drop_max_depth=self.DROP_MAX_DEPTH,
        )

        index_path = root_path / "ETH3D" / ".index.txt"
        lines = [
            ln.strip()
            for ln in index_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if scenes is not None:
            wanted = set(scenes)
            lines = [ln for ln in lines if ln.split("/", 1)[0] in wanted]
        self._records: list[dict[str, Any]] = [
            {"sample_path": ln, "sample_id": f"eth3d-moge/{ln.replace('/', '_')}"} for ln in lines
        ]
        if not self._records:
            raise DatasetNotAvailable(
                f"No ETH3D MoGe-bundle samples under {root_path}/ETH3D "
                "(empty .index.txt or scene filter matched nothing)."
            )

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        # Replicate DIODEMogeEvalLoader._load_sample: build the raw dict that
        # MoGe's ``_process_instance`` expects, run the warp, convert tensors.
        from moge.utils.io import read_depth as _moge_read_depth
        from moge.utils.io import read_image as _moge_read_image

        sample_root = self.root / "ETH3D" / rec["sample_path"]
        img_path = _find_image(sample_root)
        if img_path is None:
            raise ValueError(f"{rec['sample_id']}: no image.{{jpg,png}} in {sample_root}")
        image_np = _moge_read_image(img_path)
        depth_np = _moge_read_depth(sample_root / "depth.png")
        meta = json.loads((sample_root / "meta.json").read_text())
        raw = {
            "filename": rec["sample_path"],
            "width": self.TARGET_WIDTH,
            "height": self.TARGET_HEIGHT,
            "image": image_np,
            "depth": np.nan_to_num(depth_np, nan=1, posinf=1, neginf=1),
            "depth_mask": np.isfinite(depth_np),
            "depth_mask_inf": np.isinf(depth_np),
            "intrinsics": np.array(meta["intrinsics"], dtype=np.float32),
        }
        inst = self._moge_pipe._process_instance(raw)

        img_t = inst["image"]  # (3, H, W) float in [0, 1]
        image = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth = inst["depth"].numpy().astype(np.float32)
        valid = inst["depth_mask"].numpy().astype(bool)
        # MoGe zeros out invalid pixels; plumbline's assert_valid_depth needs
        # finite > 0, so restore a safe placeholder for masked-out pixels.
        depth_clean = np.where(valid, depth, np.float32(1.0))

        images = image[None]
        depth_gt = depth_clean[None]
        depth_valid = valid[None]
        assert_valid_image(images, name=f"{rec['sample_id']}/image")
        assert_valid_depth(depth_gt, name=f"{rec['sample_id']}/depth")

        # MoGe target intrinsics are normalized (fx/W, fy/H, cx/cy in [0,1]).
        K_norm = inst["intrinsics"].numpy().astype(np.float32)
        h, w, _ = image.shape
        K_pix = K_norm.copy()
        K_pix[0, :] *= w
        K_pix[1, :] *= h
        K_stack = K_pix[None]
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"{rec['sample_id']}/extrinsics")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": rec["sample_path"].split("/", 1)[0],
                "split": self.split,
                "source": "moge_hf_bundle",
                "sample_path": rec["sample_path"],
            },
        )
