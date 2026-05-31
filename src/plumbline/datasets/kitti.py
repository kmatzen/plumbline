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
    "KITTIMogeEvalLoader",
    "eigen_crop_mask",
    "garg_crop_mask",
    "kitti_benchmark_crop",
    "load_kitti_calib",
    "load_kitti_depth_png_to_m",
    "parse_eigen_sample_list",
]

# KITTI annotated depth PNG scale: depth_m = png_value / 256.
_DEPTH_SCALE_PNG_PER_M = 256.0

# KITTI Benchmark center crop — used by Marigold's official eval
# (``src/dataset/kitti_dataset.py::kitti_benchmark_crop`` @ prs-eth/Marigold
# HEAD) and the KITTI-depth-benchmark image size convention.
_KITTI_BM_CROP_HEIGHT = 352
_KITTI_BM_CROP_WIDTH = 1216


def kitti_benchmark_crop(
    image: NDArray[Any],
    depth: NDArray[Any],
    K: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Apply the KITTI-benchmark 1216×352 centered crop used by Marigold/
    GeoWizard/DA-V2 evaluation pipelines.

    Bottom-aligned (``top_margin = H - 352``), horizontally centered. Same
    crop is applied to the RGB image, the depth GT, and the intrinsics —
    shifting the principal point by ``(-left_margin, -top_margin)``.

    Mirrors ``KITTIDepthDataset.kitti_benchmark_crop`` in prs-eth/Marigold
    (Apache-2.0). Callers should apply this BEFORE the evaluation-time
    ``garg`` / ``eigen`` valid-mask crop so the mask fractions (0.33–0.91
    for eigen, 0.41–0.99 for garg) operate on the 352-tall benchmark
    frame rather than the raw 375-tall image.

    Returns ``(image_cropped, depth_cropped, K_adjusted)``. The image is
    HWC or HW; depth is HW. Raises ``ValueError`` if the input is smaller
    than the target crop.
    """
    H, W = image.shape[:2]
    if H < _KITTI_BM_CROP_HEIGHT or W < _KITTI_BM_CROP_WIDTH:
        raise ValueError(
            f"image too small for KITTI benchmark crop: got {H}x{W}, need "
            f">= {_KITTI_BM_CROP_HEIGHT}x{_KITTI_BM_CROP_WIDTH}"
        )
    top = H - _KITTI_BM_CROP_HEIGHT
    left = (W - _KITTI_BM_CROP_WIDTH) // 2
    img_c = image[top : top + _KITTI_BM_CROP_HEIGHT, left : left + _KITTI_BM_CROP_WIDTH]
    dep_c = depth[top : top + _KITTI_BM_CROP_HEIGHT, left : left + _KITTI_BM_CROP_WIDTH]
    K_c = K.copy()
    K_c[0, 2] -= left
    K_c[1, 2] -= top
    return img_c, dep_c, K_c


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
    apply_kitti_bm_crop
        If set, the loader applies the KITTI-benchmark 1216×352 centered
        crop (``kitti_benchmark_crop``) to the image, depth, and
        intrinsics before any valid-mask crop. This matches Marigold's
        official eval (``kitti_bm_crop: true`` in the KITTI dataset
        config) and is the right setting for paper-match on
        Marigold / GeoWizard / DA-V2 KITTI cells.
    depth_split
        ``None`` (default) scans drives whose annotated GT lives under
        either ``depth_annotated/train/`` or ``depth_annotated/val/``.
        ``"train"`` or ``"val"`` restricts to that split. The
        DUSt3R/MonST3R/CUT3R lineage evaluates on the val drives only,
        so ``protocols/kitti_dust3r_lineage.yaml`` pins ``depth_split:
        val``. Ignored when ``sample_list`` is set (the list itself is
        the authoritative selection).
    max_frames_per_drive
        If set, after sorting each drive's frames by ``frame_id``, keep
        only the first ``N``. Mirrors MonST3R's
        ``datasets_preprocess/prepare_kitti.py`` (``[:110]`` per drive)
        which is the lineage's KITTI eval-set selection. Ignored when
        ``sample_list`` is set.
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
        apply_kitti_bm_crop: bool = False,
        depth_split: str | None = None,
        max_frames_per_drive: int | None = None,
    ) -> None:
        if apply_garg_crop and apply_eigen_crop:
            raise ValueError(
                "apply_garg_crop and apply_eigen_crop are mutually exclusive; "
                "pick the one your target paper uses."
            )
        if depth_split is not None and depth_split not in ("train", "val"):
            raise ValueError(f"depth_split must be None, 'train', or 'val'; got {depth_split!r}")
        if max_frames_per_drive is not None and max_frames_per_drive <= 0:
            raise ValueError(
                f"max_frames_per_drive must be positive or None; got {max_frames_per_drive!r}"
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
        self.apply_kitti_bm_crop = bool(apply_kitti_bm_crop)
        self.depth_split = depth_split
        self.max_frames_per_drive = (
            int(max_frames_per_drive) if max_frames_per_drive is not None else None
        )

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
        if sample_list_path is not None and not sample_list_path.is_absolute():
            # Resolve relative `sample_list` in two places, in order:
            # 1. The in-repo ``reproductions/`` directory (authoritative for
            #    committed reproductions — e.g. ``kitti_eigen_benchmark_652.txt``
            #    lives there so every host evaluates the same 652 frames).
            # 2. ``$KITTI_ROOT`` (legacy, pre-2026-04: users placed the sample
            #    list alongside the data).
            # Commit-in-repo takes precedence because it eliminates silent
            # cross-host divergence.
            from plumbline.paths import REPRODUCTIONS_DIR

            repo_candidate = REPRODUCTIONS_DIR / sample_list_path
            host_candidate = self.root / sample_list_path
            sample_list_path = repo_candidate if repo_candidate.exists() else host_candidate
        list_tag = sample_list_path.stem if sample_list_path else "scan"
        # Fold the scan-mode kwargs into the manifest filename so a different
        # depth_split / per-drive cap doesn't silently reuse a stale manifest.
        if sample_list_path is None and (
            self.depth_split is not None or self.max_frames_per_drive is not None
        ):
            split_tag = self.depth_split or "all"
            cap_tag = (
                f"top{self.max_frames_per_drive}"
                if self.max_frames_per_drive is not None
                else "full"
            )
            list_tag = f"scan_{split_tag}_{cap_tag}"

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
        ``depth_annotated/val/``. When ``depth_split`` is set we restrict
        to that split; otherwise we accept whichever exists. When
        ``max_frames_per_drive`` is set we keep only the first N frames
        per drive after sorting by ``frame_id`` (mirrors MonST3R's
        ``prepare_kitti.py [:110]`` lineage-eval selection).
        """
        splits_to_try = (self.depth_split,) if self.depth_split is not None else ("train", "val")
        raw_drives = sorted(p for p in self.raw_root.rglob("*_sync") if p.is_dir())
        for drive_dir in raw_drives:
            date = drive_dir.parent.name  # e.g. 2011_09_26
            drive = drive_dir.name  # e.g. 2011_09_26_drive_0002_sync
            img_dir = drive_dir / self.camera / "data"
            if not img_dir.exists():
                continue
            # Annotated GT lives under one of the split directories.
            gt_dir = None
            for split in splits_to_try:
                candidate = (
                    self.depth_root / split / drive / "proj_depth" / "groundtruth" / self.camera
                )
                if candidate.exists():
                    gt_dir = candidate
                    break
            if gt_dir is None:
                continue
            img_paths = sorted(img_dir.glob("*.png"))
            if self.max_frames_per_drive is not None:
                # Match MonST3R / CUT3R's prepare_kitti.py: enumerate the
                # *annotated GT* frames first (sorted), take the first N,
                # then look up the matching RGB. The annotated-GT set is
                # always a subset of the raw set, so iterating GT-first
                # avoids burning the cap on raw frames that have no GT.
                gt_paths = sorted(gt_dir.glob("*.png"))[: self.max_frames_per_drive]
                kept = {p.stem for p in gt_paths}
                img_paths = [p for p in img_paths if p.stem in kept]
            for img_path in img_paths:
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
        depth = load_kitti_depth_png_to_m(self.root / rec["depth_path"])
        K = load_kitti_calib(self.root / rec["calib_path"], camera=rec["camera"])

        # KITTI Benchmark crop runs first — must happen before the
        # valid-mask crop so the crop-mask fractions operate on the 352-tall
        # frame, matching Marigold's official eval ordering.
        if self.apply_kitti_bm_crop:
            image, depth, K = kitti_benchmark_crop(image, depth, K)

        images = image[None]  # (1, H, W, 3)
        assert_valid_image(images, name=f"kitti/{rec['sample_id']}/image")

        depth_gt = depth[None]  # (1, H, W)

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
                "kitti_bm_crop": self.apply_kitti_bm_crop,
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


@register_dataset("kitti-moge-eval")
class KITTIMogeEvalLoader(Dataset):
    """KITTI loader matching MoGe's evaluation pipeline.

    Reads from the preprocessed HF bundle
    ``Ruicheng/monocular-geometry-evaluation``'s ``KITTI.zip`` — NOT the
    raw KITTI tree. MoGe's bundle ships 652 Eigen-test samples already
    center-warped (~750×375), with depth re-encoded as log-spaced 16-bit
    PNG plus a per-sample ``meta.json`` carrying normalised intrinsics.
    Same on-disk shape as ``DIODEMogeEvalLoader``; only the index path
    and the absence of a domain split differ.

    This loader closes D8 / D9 / D18 in ``docs/DISCREPANCIES.md`` — the
    structural MoGe-KITTI protocol delta that Monodepth2-Eigen + Garg +
    80 m clip doesn't capture (no crop, no cap, bespoke resolution).

    The loader delegates to MoGe's own
    ``moge.test.dataloader.EvalDataLoaderPipeline._process_instance``
    so it stays in lockstep with upstream. MoGe's eval applies a
    homographic FoV-crop from the source 1242×375 KITTI bundle down to
    the ``configs/eval/all_benchmarks.json`` KITTI target of
    ``(width=750, height=375)`` — a 2:1 aspect centered in the 3.3:1
    source, keeping ~60 % of the horizontal FoV (54.9° vs 81.4°). The
    intrinsics on the warped frame match the new FoV; depth is ray-
    length-adjusted through the warp so Z-axis depth stays consistent.
    Needs the ``moge`` package (``uv pip install 'git+https://github.com/
    microsoft/MoGe.git'``).

    Expected layout (pointed at by ``--data-root`` or ``$KITTI_MOGE_ROOT``)::

        <root>/KITTI/
          .index.txt                           # 652 sample paths, line-wise
          <sample_path>/
              image.jpg, depth.png, meta.json

    Stage via::

        hf download Ruicheng/monocular-geometry-evaluation \\
            KITTI.zip --repo-type dataset --local-dir /tmp/moge_dl
        unzip /tmp/moge_dl/KITTI.zip -d $KITTI_MOGE_ROOT

    Parameters
    ----------
    root
        Points at a directory containing the unzipped ``KITTI/`` tree.
        Falls back to ``$KITTI_MOGE_ROOT``.
    """

    # MoGe's configs/eval/all_benchmarks.json KITTI target.
    # https://github.com/microsoft/MoGe/blob/main/configs/eval/all_benchmarks.json
    TARGET_WIDTH: int = 750
    TARGET_HEIGHT: int = 375
    DEPTH_UNIT: float = 1.0
    DROP_MAX_DEPTH: float = 1000.0

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
    ) -> None:
        # Accepted for protocol-YAML compatibility; the bundle only ships
        # one split (MoGe evaluates on the Eigen test 652).
        if split != "test":
            raise ValueError(f"KITTIMogeEvalLoader only exposes the test split; got {split!r}")
        root_path = Path(root) if root else env_path("KITTI_MOGE_ROOT")
        if root_path is None or not (root_path / "KITTI").exists():
            raise DatasetNotAvailable(
                "KITTI MoGe-eval bundle not found. Set --data-root or "
                "$KITTI_MOGE_ROOT to a directory containing KITTI/.index.txt "
                "plus the unzipped KITTI/<sample_path>/... tree. Stage via: "
                "hf download Ruicheng/monocular-geometry-evaluation "
                "KITTI.zip --repo-type dataset --local-dir <tmp> && "
                "unzip <tmp>/KITTI.zip -d $KITTI_MOGE_ROOT"
            )

        self.root = root_path

        # Build a MoGe EvalDataLoaderPipeline for KITTI. Instantiation reads
        # ``.index.txt`` but doesn't launch the pipeline's worker processes
        # (that happens in .start()) — we use it as a handle to the upstream
        # ``_load_instance`` / ``_process_instance`` methods.
        try:
            from moge.test.dataloader import EvalDataLoaderPipeline
        except ModuleNotFoundError as exc:
            raise ImportError(
                "KITTIMogeEvalLoader needs the `moge` package for its "
                "homographic warp (matches the paper's eval pipeline). "
                "Install with `uv pip install "
                "'git+https://github.com/microsoft/MoGe.git'`."
            ) from exc
        self._moge_pipe = EvalDataLoaderPipeline(
            path=str(root_path / "KITTI"),
            width=self.TARGET_WIDTH,
            height=self.TARGET_HEIGHT,
            split=".index.txt",
            drop_max_depth=self.DROP_MAX_DEPTH,
            depth_unit=self.DEPTH_UNIT,
        )
        self._records: list[dict[str, Any]] = [
            {"sample_path": fn, "sample_id": fn.replace("/", "_"), "idx": i}
            for i, fn in enumerate(self._moge_pipe.filenames)
        ]

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        # Build the input dict that MoGe's ``_process_instance`` expects.
        # We don't call its ``_load_instance`` because upstream references an
        # undefined ``read_meta`` symbol (stale after a refactor — raw bug in
        # MoGe HEAD). The build below replicates ``_load_instance`` exactly.
        import json as _json

        from moge.utils.io import read_depth as _moge_read_depth
        from moge.utils.io import read_image as _moge_read_image

        sample_root = self.root / "KITTI" / rec["sample_path"]
        image_np = _moge_read_image(sample_root / "image.jpg")
        depth_np = _moge_read_depth(sample_root / "depth.png")
        meta = _json.loads((sample_root / "meta.json").read_text())
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

        # MoGe returns torch tensors in CHW / HWC. Plumbline wants
        # uint8 HWC image + float32 HW depth + bool HW mask as numpy.
        img_t = inst["image"]  # (3, H, W) float in [0,1]
        image = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth = inst["depth"].numpy().astype(np.float32)
        valid = inst["depth_mask"].numpy().astype(bool)
        # MoGe zeros out invalid pixels (line 169 of dataloader.py); plumbline
        # uses 1.0 as a safe placeholder so depth-as-metric tensors never hit
        # 0 or NaN. Restore that convention here.
        depth_clean = np.where(valid, depth, np.float32(1.0))

        images = image[None]
        depth_gt = depth_clean[None]
        depth_valid = valid[None]
        assert_valid_image(images, name=f"kitti_moge/{rec['sample_id']}/image")
        assert_valid_depth(depth_gt, name=f"kitti_moge/{rec['sample_id']}/depth")

        # MoGe's target intrinsics are normalized in its own convention:
        # fx_norm = fx_pix / W, fy_norm = fy_pix / H, cx/cy in [0,1].
        K_norm = inst["intrinsics"].numpy().astype(np.float32)
        h, w, _ = image.shape
        K_pix = K_norm.copy()
        K_pix[0, :] *= w
        K_pix[1, :] *= h
        K_stack = K_pix[None]
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"kitti_moge/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"kitti_moge/{rec['sample_id']}/extrinsics")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "split": self.split,
                "source": "moge_hf_bundle",
                "sample_path": rec["sample_path"],
            },
        )
