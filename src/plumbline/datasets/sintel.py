"""MPI-Sintel loader.

Sintel is a synthetic rendered-animation dataset. It includes per-frame depth,
flow, camera intrinsics, and extrinsics. In plumbline we use the ``training``
split (it's the only one with GT depth; there is no public test GT).

Expected layout (point ``--data-root`` or ``$SINTEL_ROOT`` at this)::

    <root>/
      training/
        final/<scene>/frame_XXXX.png
        clean/<scene>/frame_XXXX.png
        depth/<scene>/frame_XXXX.dpt
        camdata_left/<scene>/frame_XXXX.cam

Download instructions: http://sintel.is.tue.mpg.de/ (MPI-Sintel stereo +
camera + depth archives are separate files).

Conventions
-----------
- ``.dpt`` files hold depth in meters. Big-endian ``float32``, little-endian
  header. We parse them directly (see :func:`_load_dpt`).
- ``.cam`` files hold ``K`` then ``P = K @ [R | t]`` as ``camera_from_world``
  projection matrix. We decompose to ``(K, R, t)`` then invert to
  ``world_from_camera``, then rebase so the first camera is identity.
"""

from __future__ import annotations

import struct
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
    invert_pose,
    rebase_to_first_camera,
)
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    load_manifest,
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["SintelDataset", "load_cam", "load_dpt"]


@register_dataset("sintel")
class SintelDataset(Dataset):
    """MPI-Sintel dataset loader.

    Parameters
    ----------
    root
        Dataset root (``<root>/training/...``). If omitted, falls back to
        ``$SINTEL_ROOT``.
    split
        Currently only ``"training"`` is supported; no public GT for test.
    pass_name
        ``"final"`` (with motion blur, atmosphere) or ``"clean"`` (no
        post-processing). Benchmarks use ``"final"`` by default.
    scenes
        Optional list of scene names to restrict to. Default: all.
    views_per_sample
        Number of consecutive frames grouped into one :class:`Sample`. ``1``
        = monocular (default); ``>1`` = sequential multi-view.
    max_depth
        If set, the loader writes ``Sample.depth_valid`` = ``(gt > 0) & (gt
        < max_depth)``. Sintel encodes sky as ~1e5 m in the ``.dpt`` file
        (it has no true infinity sentinel); the DUSt3R / MonST3R / CUT3R
        eval lineage masks sky by clipping to ``max_depth=70`` (per
        MonST3R's ``depth_metric.ipynb`` Sintel cell). Leave ``None`` to
        pass the raw GT through (any 1e5 sky pixels will dominate
        AbsRel, so this is **only** correct for adapters / protocols that
        handle sky themselves).
    """

    split: str = "training"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "training",
        pass_name: str = "final",
        scenes: list[str] | None = None,
        views_per_sample: int = 1,
        max_depth: float | None = None,
    ) -> None:
        if split != "training":
            raise ValueError(f"Sintel split '{split}' has no public GT; use 'training'")
        if max_depth is not None and max_depth <= 0:
            raise ValueError(f"max_depth must be positive or None; got {max_depth!r}")
        root_path = Path(root) if root else env_path("SINTEL_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "Sintel not found. Set --data-root or $SINTEL_ROOT to a directory "
                "containing training/<final|clean|depth|camdata_left>/<scene>/*. "
                "Download from http://sintel.is.tue.mpg.de/."
            )
        self.root = root_path
        self.split = split
        self.pass_name = pass_name
        self.views_per_sample = max(1, int(views_per_sample))
        self.max_depth = float(max_depth) if max_depth is not None else None

        manifest_path = (
            self.root
            / ".plumbline_manifest"
            / f"sintel_{split}_{pass_name}_vps{self.views_per_sample}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan(pass_name, scenes))
            save_manifest(manifest_path, records)
        if scenes:
            records = [r for r in records if r["scene"] in scenes]
        self._records = records

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self, pass_name: str, scenes: list[str] | None) -> Iterator[dict[str, Any]]:
        images_root = self.root / "training" / pass_name
        depth_root = self.root / "training" / "depth"
        cam_root = self.root / "training" / "camdata_left"
        if not images_root.exists():
            raise DatasetNotAvailable(f"Sintel pass '{pass_name}' not found at {images_root}")
        scene_dirs = sorted(p for p in images_root.iterdir() if p.is_dir())
        if scenes is not None:
            wanted = set(scenes)
            scene_dirs = [p for p in scene_dirs if p.name in wanted]
        for scene_dir in scene_dirs:
            frames = sorted(scene_dir.glob("frame_*.png"))
            for i in range(0, len(frames) - self.views_per_sample + 1):
                view_frames = frames[i : i + self.views_per_sample]
                rec = {
                    "sample_id": f"{scene_dir.name}/{view_frames[0].stem}_v{self.views_per_sample}",
                    "scene": scene_dir.name,
                    "image_paths": [str(f.relative_to(self.root)) for f in view_frames],
                    "depth_paths": [
                        str(
                            (depth_root / scene_dir.name / (f.stem + ".dpt")).relative_to(self.root)
                        )
                        for f in view_frames
                    ],
                    "cam_paths": [
                        str((cam_root / scene_dir.name / (f.stem + ".cam")).relative_to(self.root))
                        for f in view_frames
                    ],
                }
                yield rec

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        images = np.stack([read_rgb_uint8(self.root / p) for p in rec["image_paths"]], axis=0)
        assert_valid_image(images, name=f"sintel/{rec['sample_id']}/image")

        Ks = []
        Es_world_from_cam = []
        depths = []
        for i, cam_rel in enumerate(rec["cam_paths"]):
            K, E_cam_from_world = load_cam(self.root / cam_rel)
            Ks.append(K)
            Es_world_from_cam.append(invert_pose(E_cam_from_world))
            depth = load_dpt(self.root / rec["depth_paths"][i])
            depths.append(depth)
        intrinsics = np.stack(Ks).astype(np.float32)
        extrinsics = rebase_to_first_camera(np.stack(Es_world_from_cam).astype(np.float64)).astype(
            np.float32
        )
        depth_gt = np.stack(depths).astype(np.float32)

        assert_valid_intrinsics(intrinsics, name=f"sintel/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(extrinsics, name=f"sintel/{rec['sample_id']}/extrinsics")
        assert_valid_depth(depth_gt, name=f"sintel/{rec['sample_id']}/depth")

        depth_valid: NDArray[np.bool_] | None = None
        if self.max_depth is not None:
            # Sintel encodes sky as ~1e5 m (no infinity sentinel); the DUSt3R
            # / MonST3R lineage masks it with max_depth (typically 70 m).
            depth_valid = (depth_gt > 0) & (depth_gt < self.max_depth)

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": rec["scene"],
                "pass": self.pass_name,
                "split": self.split,
                "max_depth": self.max_depth,
            },
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------

_DPT_TAG = 202021.25  # .dpt / .flo shared magic number


def load_dpt(path: Path) -> NDArray[np.float32]:
    """Parse Sintel ``.dpt`` depth (little-endian, shared with ``.flo``).

    Format (from Sintel SDK ``sintel_io.py``):
    - float32 tag (``TAG_FLOAT`` = 202021.25)
    - int32 width
    - int32 height
    - H*W float32 depth values
    """
    with path.open("rb") as f:
        tag = struct.unpack("<f", f.read(4))[0]
        if not np.isclose(tag, _DPT_TAG, atol=1e-3):
            raise ValueError(f"bad .dpt tag in {path}: {tag}")
        w, h = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(4 * w * h), dtype="<f4").reshape(h, w)
    return np.ascontiguousarray(data)


def load_cam(path: Path) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Parse a Sintel ``.cam`` file.

    Format (from Sintel SDK ``sintel_io.cam_read``):
    - float32 tag (``TAG_FLOAT`` = 202021.25, same magic as ``.dpt`` / ``.flo``)
    - float64 3x3 intrinsics ``K`` (row-major)
    - float64 3x4 extrinsics ``[R|t]`` (``camera_from_world``)

    Returns ``(K, E_cam_from_world)`` where ``E`` is the 4x4 homogenized
    extrinsic matrix. The tag (4 bytes) was previously skipped — without
    reading it, K was misaligned by 4 bytes and the bottom row of K parsed
    to zeros, tripping ``assert_valid_intrinsics``. Real on-disk file size
    is 4 + 8*9 + 8*12 = 172 bytes.
    """
    with path.open("rb") as f:
        tag = struct.unpack("<f", f.read(4))[0]
        if not np.isclose(tag, _DPT_TAG, atol=1e-3):
            raise ValueError(f"bad .cam tag in {path}: {tag}")
        K = np.frombuffer(f.read(8 * 9), dtype="<f8").reshape(3, 3)
        RT = np.frombuffer(f.read(8 * 12), dtype="<f8").reshape(3, 4)
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = RT
    return K.astype(np.float32), E.astype(np.float32)
