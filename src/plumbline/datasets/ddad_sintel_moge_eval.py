"""DDAD and Sintel mono-depth loaders via MoGe's preprocessed eval bundle.

Both datasets ship in ``Ruicheng/monocular-geometry-evaluation`` (DDAD.zip,
Sintel.zip) with the same nested ``.index.txt`` layout as ETH3D/DIODE MoGe
bundles. Evaluation uses MoGe Table 3's **affine-invariant disparity** column
(Wang et al. 2024, arXiv:2410.19115).

Distinct from:

- ``sintel`` — native MPI-Sintel tree for MonST3R / Depth Pro / DA-V2 Table 2.
- ``kitti`` / raw driving sets — different preprocessing.

Warp sizes from MoGe ``configs/eval/all_benchmarks.json``:

- DDAD: 1400×700
- Sintel: 872×436 (center crop, sky masked in bundle)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
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

__all__ = [
    "DDADMogeEvalDataset",
    "SintelMogeEvalDataset",
]


def _find_image(sample_dir: Path) -> Path | None:
    for ext in ("jpg", "png", "jpeg"):
        p = sample_dir / f"image.{ext}"
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class _MogeIndexBundleSpec:
    bundle_subdir: str
    env_var: str
    target_width: int
    target_height: int
    id_prefix: str
    display_name: str


class _MogeIndexBundleDataset(Dataset):
    """Shared loader for MoGe HF bundles with ``<Bundle>/.index.txt`` trees."""

    split: str = "test"

    def __init__(
        self,
        *,
        spec: _MogeIndexBundleSpec,
        root: Path | str | None = None,
        split: str = "test",
    ) -> None:
        if split != "test":
            raise ValueError(f"{spec.display_name} only exposes the test split; got {split!r}")
        root_path = Path(root) if root else env_path(spec.env_var)
        bundle_root = root_path / spec.bundle_subdir if root_path is not None else None
        if root_path is None or bundle_root is None or not bundle_root.exists():
            raise DatasetNotAvailable(
                f"{spec.display_name} MoGe bundle not found. Set --data-root or "
                f"${spec.env_var} to a directory containing "
                f"{spec.bundle_subdir}/.index.txt plus the unzipped "
                f"{spec.bundle_subdir}/<path>/{{image,depth,meta}} tree. "
                "Stage via: hf download Ruicheng/monocular-geometry-evaluation "
                f"--repo-type dataset --include '{spec.bundle_subdir}*' "
                "--local-dir <tmp> && "
                f"unzip <tmp>/{spec.bundle_subdir}.zip -d ${spec.env_var}"
            )
        self.spec = spec
        self.root = root_path

        try:
            from plumbline.models.moge import _ensure_moge_on_path

            _ensure_moge_on_path()  # vendored moge + its pinned utils3d/pipeline
            from moge.test.dataloader import EvalDataLoaderPipeline
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"{spec.display_name} needs the vendored `moge` eval pipeline "
                "(plumbline/_vendor/moge — the homographic warp matching the "
                "paper's eval). The install is likely corrupt; reinstall "
                "plumbline, or set $MOGE_ROOT to an upstream MoGe checkout."
            ) from exc
        self._moge_pipe = EvalDataLoaderPipeline(
            path=str(bundle_root),
            width=spec.target_width,
            height=spec.target_height,
            split=".index.txt",
            drop_max_depth=1000.0,
        )

        index_path = bundle_root / ".index.txt"
        lines = [
            ln.strip()
            for ln in index_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        self._records: list[dict[str, Any]] = [
            {
                "sample_path": ln,
                "sample_id": f"{spec.id_prefix}/{ln.replace('/', '_')}",
            }
            for ln in lines
        ]
        if not self._records:
            raise DatasetNotAvailable(
                f"No {spec.display_name} samples under {bundle_root} (empty .index.txt)."
            )

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        from moge.utils.io import read_depth as _moge_read_depth
        from moge.utils.io import read_image as _moge_read_image

        sample_root = self.root / self.spec.bundle_subdir / rec["sample_path"]
        img_path = _find_image(sample_root)
        if img_path is None:
            raise ValueError(f"{rec['sample_id']}: no image.{{jpg,png}} in {sample_root}")
        image_np = _moge_read_image(img_path)
        depth_np = _moge_read_depth(sample_root / "depth.png")
        meta = json.loads((sample_root / "meta.json").read_text())
        raw = {
            "filename": rec["sample_path"],
            "width": self.spec.target_width,
            "height": self.spec.target_height,
            "image": image_np,
            "depth": np.nan_to_num(depth_np, nan=1, posinf=1, neginf=1),
            "depth_mask": np.isfinite(depth_np),
            "depth_mask_inf": np.isinf(depth_np),
            "intrinsics": np.array(meta["intrinsics"], dtype=np.float32),
        }
        inst = self._moge_pipe._process_instance(raw)

        img_t = inst["image"]
        image = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth = inst["depth"].numpy().astype(np.float32)
        valid = inst["depth_mask"].numpy().astype(bool)
        depth_clean = np.where(valid, depth, np.float32(1.0))

        images = image[None]
        depth_gt = depth_clean[None]
        depth_valid = valid[None]
        assert_valid_image(images, name=f"{rec['sample_id']}/image")
        assert_valid_depth(depth_gt, name=f"{rec['sample_id']}/depth")

        K_norm = inst["intrinsics"].numpy().astype(np.float32)
        h, w, _ = image.shape
        K_pix = K_norm.copy()
        K_pix[0, :] *= w
        K_pix[1, :] *= h
        K_stack = K_pix[None]
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"{rec['sample_id']}/extrinsics")

        parts = rec["sample_path"].split("/", 1)
        scene = parts[0] if len(parts) > 1 else rec["sample_path"]

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": scene,
                "split": self.split,
                "source": "moge_hf_bundle",
                "sample_path": rec["sample_path"],
            },
        )


_DDAD_SPEC = _MogeIndexBundleSpec(
    bundle_subdir="DDAD",
    env_var="DDAD_MOGE_ROOT",
    target_width=1400,
    target_height=700,
    id_prefix="ddad-moge",
    display_name="DDAD",
)


_SINTEL_SPEC = _MogeIndexBundleSpec(
    bundle_subdir="Sintel",
    env_var="SINTEL_MOGE_ROOT",
    target_width=872,
    target_height=436,
    id_prefix="sintel-moge",
    display_name="Sintel",
)


@register_dataset("ddad-moge-eval")
class DDADMogeEvalDataset(_MogeIndexBundleDataset):
    """DDAD single-frame mono-depth from MoGe's eval bundle (1000 val samples)."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
    ) -> None:
        super().__init__(spec=_DDAD_SPEC, root=root, split=split)


@register_dataset("sintel-moge-eval")
class SintelMogeEvalDataset(_MogeIndexBundleDataset):
    """Sintel single-frame mono-depth from MoGe's eval bundle (1064 frames)."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
    ) -> None:
        super().__init__(spec=_SINTEL_SPEC, root=root, split=split)
