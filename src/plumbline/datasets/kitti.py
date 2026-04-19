"""KITTI depth-evaluation loader (Eigen split, annotated-GT protocol).

KITTI is the canonical *outdoor* monocular-depth benchmark. The "Eigen test
split" (Eigen et al. 2014, NIPS) selects ~697 images from the KITTI raw
recordings and is evaluated against sparse LiDAR-reprojected depth, or —
more recently — against the KITTI Depth-Prediction Benchmark's *annotated*
dense depth maps (Uhrig et al. 2017, 3DV). DA-V2, Metric3Dv2, DA3, MoGe,
Depth Pro, and every modern mono-depth paper report numbers with the
annotated GT on the 652-image "with-GT" subset of the Eigen test split.

Expected layout::

    <root>/
      raw/                                    # KITTI raw sequences
        2011_09_26/
          calib_cam_to_cam.txt
          calib_velo_to_cam.txt
          2011_09_26_drive_0002_sync/
            image_02/data/0000000069.png      # left rectified color
            image_03/data/0000000069.png      # right rectified color (optional)
      depth_annotated/                        # KITTI Depth Benchmark GT
        train/ val/
          2011_09_26_drive_0002_sync/
            proj_depth/groundtruth/image_02/0000000069.png   # uint16 PNG

Download (public, no ToS):

- Raw sequences: https://www.cvlibs.net/datasets/kitti/raw_data.php
  (download the synced+rectified archives for the drives in your sample
  list, plus each drive's ``calib`` archive).
- Annotated depth maps: https://www.cvlibs.net/datasets/kitti/eval_depth.php
  (``data_depth_annotated.zip``, ~14 GB — free, no account).

The canonical Eigen test-split sample lists are not bundled because
multiple variants circulate (697 original Eigen 2014; 652 with-GT; 500
improved). Get one from the Monodepth2 repo at
https://github.com/nianticlabs/monodepth2/tree/master/splits/eigen or
equivalent, and pass it as ``sample_list``. Without a list, the loader
scans for every ``(image, depth)`` pair it can pair up and iterates them
sorted.

Conventions
-----------
- Images are (H, W, 3) uint8 sRGB. KITTI raw images are typically
  (375, 1242); individual drives vary slightly.
- Depth is (H, W) float32 meters. KITTI annotated depth PNGs are uint16
  encoded as ``depth_m = png_value / 256.0``; ``png_value == 0`` means
  invalid. This loader returns invalid pixels as ``0`` (our convention).
- Intrinsics come from ``calib_cam_to_cam.txt`` and depend on which rect-
  ified camera is used. For ``image_02`` (left color) the relevant matrix
  is ``P_rect_02`` (3x4 projection); we take its upper-left 3x3 as K.
  Every drive under the same date shares a single calibration file.
- KITTI is a mono depth benchmark (each sample is one view). Extrinsics
  are identity.

Evaluation cropping
-------------------
Two crops appear across papers; use whichever the paper specifies. Pass
``apply_garg_crop=True`` or ``apply_eigen_crop=True`` to have the loader
write ``Sample.depth_valid`` = (GT > 0) ∧ (crop mask). The standalone
:func:`garg_crop_mask` / :func:`eigen_crop_mask` helpers are also
available for callers that want to apply a crop outside the loader.

- **Garg crop** (Garg et al. 2016): evaluate only pixels with row in
  ``[0.40810811 * H, 0.99189189 * H)`` and col in ``[0.03594771 * W,
  0.96405229 * W)``. Used by Monodepth, DA-V2, most self-supervised work.
- **Eigen crop** (Eigen et al. 2014): row ``[0.3324 * H, 0.91351351 * H)``
  and col ``[0.0359477 * W, 0.96405229 * W)``. Used by the original Eigen
  paper and some supervised work; overlaps substantially with Garg but
  neither is a strict superset.
"""

from __future__ import annotations

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
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    load_manifest,
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = [
    "KITTIDataset",
    "eigen_crop_mask",
    "garg_crop_mask",
    "load_kitti_calib",
    "load_kitti_depth_png_to_m",
    "parse_eigen_sample_list",
]

# KITTI annotated depth PNG scale: depth_m = png_value / 256.
_DEPTH_SCALE_PNG_PER_M = 256.0


@register_dataset("kitti")
class KITTIDataset(Dataset):
    """KITTI Eigen-split depth-evaluation loader.

    Parameters
    ----------
    root
        Dataset root containing ``raw/<date>/...`` and
        ``depth_annotated/<train|val>/...``. If omitted, falls back to
        ``$KITTI_ROOT``.
    raw_subdir, depth_subdir
        Relative directory names under ``root``. Defaults match the layout
        documented in the module docstring. Override if you've unpacked the
        archives differently.
    sample_list
        Optional path to an Eigen-style sample list. Each non-empty,
        non-comment line is parsed with :func:`parse_eigen_sample_list`.
        When omitted, the loader scans for all ``(image, depth)`` pairs it
        can match across every drive under ``raw_subdir`` and iterates them
        in sorted order. For paper reproductions, always pass an explicit
        sample list.
    camera
        ``"image_02"`` (left rectified color, default) or ``"image_03"``
        (right). Matches the token in Eigen-style sample lines.
    apply_garg_crop, apply_eigen_crop
        If set, the loader writes ``Sample.depth_valid`` = (GT > 0) ∧ (crop
        mask). Mirrors the NYUv2 loader's ``apply_eigen_crop`` kwarg so a
        reproduction YAML can opt in with a single flag. The two are
        mutually exclusive; set at most one.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        raw_subdir: str = "raw",
        depth_subdir: str = "depth_annotated",
        sample_list: Path | str | None = None,
        camera: str = "image_02",
        apply_garg_crop: bool = False,
        apply_eigen_crop: bool = False,
    ) -> None:
        if apply_garg_crop and apply_eigen_crop:
            raise ValueError(
                "apply_garg_crop and apply_eigen_crop are mutually exclusive; "
                "pick the one your target paper uses."
            )
        root_path = Path(root) if root else env_path("KITTI_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "KITTI not found. Set --data-root or $KITTI_ROOT to a directory "
                "containing raw/<date>/<drive>_sync/image_0{2,3}/data/*.png plus "
                "depth_annotated/<split>/<drive>_sync/proj_depth/groundtruth/image_0{2,3}/*.png. "
                "Downloads (both public, no account): "
                "https://www.cvlibs.net/datasets/kitti/raw_data.php ; "
                "https://www.cvlibs.net/datasets/kitti/eval_depth.php "
                "(data_depth_annotated.zip)."
            )
        if camera not in ("image_02", "image_03"):
            raise ValueError(f"camera must be 'image_02' or 'image_03'; got {camera!r}")

        self.root = root_path
        self.raw_root = root_path / raw_subdir
        self.depth_root = root_path / depth_subdir
        self.camera = camera
        self.apply_garg_crop = bool(apply_garg_crop)
        self.apply_eigen_crop = bool(apply_eigen_crop)

        if not self.raw_root.exists():
            raise DatasetNotAvailable(
                f"KITTI raw tree not found at {self.raw_root}. Expected the synced+rectified "
                "drives unpacked under this path (see module docstring)."
            )
        if not self.depth_root.exists():
            raise DatasetNotAvailable(
                f"KITTI annotated-depth tree not found at {self.depth_root}. Unpack "
                "data_depth_annotated.zip so that <drive>_sync/proj_depth/groundtruth/* lives "
                "under this path."
            )

        sample_list_path = Path(sample_list) if sample_list else None
        list_tag = sample_list_path.stem if sample_list_path else "scan"

        manifest_path = self.root / ".plumbline_manifest" / f"kitti_{list_tag}_{camera}.jsonl"
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            if sample_list_path is not None:
                if not sample_list_path.exists():
                    raise DatasetNotAvailable(f"sample_list not found: {sample_list_path}")
                entries = parse_eigen_sample_list(sample_list_path)
                records = list(self._records_from_entries(entries))
            else:
                records = list(self._scan_pairs())
            save_manifest(manifest_path, records)
        self._records = records

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan_pairs(self) -> Iterator[dict[str, Any]]:
        """Iterate every image with a matching annotated-depth GT.

        Drives can appear under either ``depth_annotated/train/`` or
        ``depth_annotated/val/``; we try both.
        """
        raw_drives = sorted(p for p in self.raw_root.rglob("*_sync") if p.is_dir())
        for drive_dir in raw_drives:
            date = drive_dir.parent.name  # e.g. 2011_09_26
            drive = drive_dir.name  # e.g. 2011_09_26_drive_0002_sync
            img_dir = drive_dir / self.camera / "data"
            if not img_dir.exists():
                continue
            # Annotated GT lives under either split directory.
            gt_dir = None
            for split in ("train", "val"):
                candidate = (
                    self.depth_root / split / drive / "proj_depth" / "groundtruth" / self.camera
                )
                if candidate.exists():
                    gt_dir = candidate
                    break
            if gt_dir is None:
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                frame_id = img_path.stem
                depth_path = gt_dir / f"{frame_id}.png"
                if not depth_path.exists():
                    continue
                yield _make_record(
                    date=date,
                    drive=drive,
                    frame_id=frame_id,
                    camera=self.camera,
                    image_path=img_path,
                    depth_path=depth_path,
                    calib_path=self.raw_root / date / "calib_cam_to_cam.txt",
                    root=self.root,
                )

    def _records_from_entries(
        self, entries: list[tuple[str, str, str]]
    ) -> Iterator[dict[str, Any]]:
        """Yield records for each ``(drive, frame_id, camera)`` entry.

        An entry's ``camera`` token must match ``self.camera`` exactly; we
        don't silently re-route across cameras since they have different
        intrinsics and a paper's number is tied to a specific camera.
        """
        for drive, frame_id, cam in entries:
            if cam != self.camera:
                raise ValueError(
                    f"sample_list entry '{drive} {frame_id} {cam}' does not match "
                    f"camera={self.camera!r}. Pass camera={cam!r} or regenerate the list."
                )
            date = _date_from_drive(drive)
            img_path = self.raw_root / date / drive / cam / "data" / f"{frame_id}.png"
            # Depth may live under either train/ or val/; find whichever exists.
            depth_path: Path | None = None
            for split in ("train", "val"):
                candidate = (
                    self.depth_root
                    / split
                    / drive
                    / "proj_depth"
                    / "groundtruth"
                    / cam
                    / f"{frame_id}.png"
                )
                if candidate.exists():
                    depth_path = candidate
                    break
            if depth_path is None or not img_path.exists():
                raise DatasetNotAvailable(
                    f"Missing data for sample {drive}/{frame_id} {cam}: "
                    f"image_exists={img_path.exists()} depth_path={depth_path}. "
                    "Unpack both the raw drive archive and data_depth_annotated.zip."
                )
            yield _make_record(
                date=date,
                drive=drive,
                frame_id=frame_id,
                camera=cam,
                image_path=img_path,
                depth_path=depth_path,
                calib_path=self.raw_root / date / "calib_cam_to_cam.txt",
                root=self.root,
            )

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        image = read_rgb_uint8(self.root / rec["image_path"])
        images = image[None]  # (1, H, W, 3)
        assert_valid_image(images, name=f"kitti/{rec['sample_id']}/image")

        depth = load_kitti_depth_png_to_m(self.root / rec["depth_path"])
        depth_gt = depth[None]  # (1, H, W)

        K = load_kitti_calib(self.root / rec["calib_path"], camera=rec["camera"])
        K_stack = K[None].astype(np.float32)  # (1, 3, 3)
        E_eye = np.eye(4, dtype=np.float32)[None]  # (1, 4, 4)

        assert_valid_intrinsics(K_stack, name=f"kitti/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"kitti/{rec['sample_id']}/extrinsics")
        assert_valid_depth(depth_gt, name=f"kitti/{rec['sample_id']}/depth")

        depth_valid: NDArray[np.bool_] | None = None
        if self.apply_garg_crop or self.apply_eigen_crop:
            crop = (
                garg_crop_mask(depth.shape)
                if self.apply_garg_crop
                else eigen_crop_mask(depth.shape)
            )
            mask = crop & (depth > 0)
            depth_valid = mask[None]

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "date": rec["date"],
                "drive": rec["drive"],
                "frame_id": rec["frame_id"],
                "camera": rec["camera"],
                "depth_scale_png_per_m": _DEPTH_SCALE_PNG_PER_M,
                "crop": (
                    "garg" if self.apply_garg_crop else ("eigen" if self.apply_eigen_crop else None)
                ),
            },
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------


def load_kitti_depth_png_to_m(path: Path) -> NDArray[np.float32]:
    """Load a KITTI annotated-depth PNG (uint16) and convert to float32 meters.

    KITTI encodes depth as ``png_value = round(depth_m * 256)``; ``png_value
    == 0`` means invalid (no LiDAR return accumulated). We return invalid
    pixels as exact ``0.0``, matching plumbline's convention.
    """
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.dtype != np.uint16:
        raise ValueError(
            f"expected uint16 depth from {path}, got {arr.dtype}. "
            "Make sure you unpacked data_depth_annotated.zip (not the velodyne "
            "LiDAR scans)."
        )
    depth = arr.astype(np.float32) / _DEPTH_SCALE_PNG_PER_M
    return depth


def load_kitti_calib(path: Path, *, camera: str = "image_02") -> NDArray[np.float32]:
    """Load intrinsics ``K`` (3x3, float32, pixels) for a KITTI rectified camera.

    Parses ``calib_cam_to_cam.txt`` and returns the upper-left 3x3 of the
    rectified projection matrix ``P_rect_<NN>`` for ``image_0N``. Every drive
    under the same date shares a single calibration file.
    """
    key = "P_rect_" + camera.split("_")[-1]  # image_02 -> P_rect_02
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name, _, rest = line.partition(":")
            if name.strip() == key:
                values = [float(v) for v in rest.split()]
                if len(values) != 12:
                    raise ValueError(f"{path}: expected 12 values for {key}, got {len(values)}")
                P = np.asarray(values, dtype=np.float64).reshape(3, 4)
                K = P[:, :3].astype(np.float32)
                return K
    raise ValueError(f"{path}: no '{key}:' line found; is this a KITTI calib file?")


def parse_eigen_sample_list(path: Path) -> list[tuple[str, str, str]]:
    """Parse an Eigen-style KITTI sample list.

    Two common formats are supported; each non-empty, non-comment line is
    either of:

    1. ``<date>/<drive>_sync <frame_id> <l|r>`` — the Monodepth2 /
       eigen_zhou split convention. ``l`` maps to ``image_02`` and ``r``
       to ``image_03``.
    2. ``<drive>_sync <frame_id> <image_0N>`` — explicit camera name.

    Returns a list of ``(drive, frame_id, camera)`` triples where
    ``camera`` is canonicalized to ``image_02``/``image_03`` and
    ``frame_id`` is zero-padded to 10 digits (the KITTI file-naming
    convention).
    """
    out: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}: cannot parse '{line}' — expected 3 whitespace fields")
            drive_token, frame_token, cam_token = parts[0], parts[1], parts[2]
            # Drive token may be "<date>/<drive>_sync" or just "<drive>_sync".
            drive = drive_token.split("/")[-1]
            if not drive.endswith("_sync"):
                raise ValueError(f"{path}: drive token '{drive_token}' does not end in '_sync'")
            camera = _canonical_camera(cam_token)
            frame_id = _canonical_frame_id(frame_token)
            out.append((drive, frame_id, camera))
    if not out:
        raise ValueError(f"{path}: parsed 0 sample entries")
    return out


# ---------------------------------------------------------------------------
# Evaluation crops
# ---------------------------------------------------------------------------


def garg_crop_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Garg et al. 2016 KITTI evaluation crop.

    Parameters
    ----------
    shape
        ``(H, W)`` in pixels. The fractional crop is applied per-image, so
        both KITTI's typical 375x1242 and off-sized frames work.

    Returns
    -------
    mask
        ``(H, W)`` boolean; True where evaluation should occur.
    """
    h, w = shape
    top = round(0.40810811 * h)
    bot = round(0.99189189 * h)
    left = round(0.03594771 * w)
    right = round(0.96405229 * w)
    mask = np.zeros(shape, dtype=bool)
    mask[top:bot, left:right] = True
    return mask


def eigen_crop_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Eigen et al. 2014 KITTI evaluation crop.

    Overlaps substantially with :func:`garg_crop_mask` but neither is a
    strict superset: Eigen's top is higher (0.3324 vs 0.408), Garg's bottom
    is lower (0.992 vs 0.914).
    """
    h, w = shape
    top = round(0.3324 * h)
    bot = round(0.91351351 * h)
    left = round(0.0359477 * w)
    right = round(0.96405229 * w)
    mask = np.zeros(shape, dtype=bool)
    mask[top:bot, left:right] = True
    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_from_drive(drive: str) -> str:
    """``'2011_09_26_drive_0002_sync'`` -> ``'2011_09_26'``."""
    # KITTI drive names are ``<yyyy>_<mm>_<dd>_drive_<NNNN>_sync``. Take the
    # first three underscore-separated tokens.
    parts = drive.split("_")
    if len(parts) < 4:
        raise ValueError(f"cannot extract date from drive {drive!r}")
    return "_".join(parts[:3])


def _canonical_camera(token: str) -> str:
    if token == "l" or token == "image_02":
        return "image_02"
    if token == "r" or token == "image_03":
        return "image_03"
    raise ValueError(f"unknown camera token {token!r}; expected l/r or image_02/image_03")


def _canonical_frame_id(token: str) -> str:
    # KITTI frame files are zero-padded to 10 digits; some sample lists are not.
    if not token.isdigit():
        raise ValueError(f"frame id must be numeric; got {token!r}")
    return token.zfill(10)


def _make_record(
    *,
    date: str,
    drive: str,
    frame_id: str,
    camera: str,
    image_path: Path,
    depth_path: Path,
    calib_path: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        "sample_id": f"{drive}/{frame_id}/{camera}",
        "date": date,
        "drive": drive,
        "frame_id": frame_id,
        "camera": camera,
        "image_path": str(image_path.relative_to(root)),
        "depth_path": str(depth_path.relative_to(root)),
        "calib_path": str(calib_path.relative_to(root)),
    }
