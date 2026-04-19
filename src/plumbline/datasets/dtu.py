"""DTU MVS dataset loader (MVSNet-repacked layout).

DTU (Jensen et al. 2014) is the canonical dense MVS benchmark — a
controlled indoor rig with ~49 rectified views per scan and a GT
laser-scanned point cloud per scan. VGGT's Table 2 (Overall chamfer =
0.382), MASt3R, DUSt3R, MVSNet, Vis-MVSNet, and basically every
learning-based MVS paper reports numbers on the standard 22-scan test
subset. The protocol VGGT follows comes from MASt3R (§4.2 of the VGGT
paper).

Expected layout (MVSNet-repacked, the de facto community format)::

    <root>/
      Cameras_1/
        00000000_cam.txt          # view 0 calibration (shared across scans)
        00000001_cam.txt          # ...
        00000048_cam.txt          # 49 views, 0-indexed here
      Rectified/
        scan1_train/
          rect_001_3_r5000.png    # view 001 (1-indexed in filename), light 3
          rect_001_0_r5000.png    # ... 7 lighting conditions 0..6
        scan4_train/
          ...
      Points/stl/
        stl001_total.ply          # GT laser scan for scan 1
        stl004_total.ply

Download (public, no ToS; MVSNet format):
- MVSNet's preprocessed DTU training split (~115 GB):
  https://roboimagedata.compute.dtu.dk/?page_id=36 +
  https://github.com/YoYo000/MVSNet
- Or the smaller test-only repack used by MASt3R / DUSt3R repos. Either
  works; the loader scans whatever is under ``Rectified/`` and picks up
  matching GT from ``Points/stl/``.

Conventions
-----------
- Images are (1200, 1600, 3) uint8 sRGB at MVSNet's native rectified size.
- Cam files store ``cam_from_world`` extrinsics (OpenCV). The loader
  inverts to ``world_from_camera`` and rebases to first-camera-as-world
  (plumbline's canonical frame).
- Intrinsics come from each view's cam file; all scans under a given
  MVSNet dump share the ``Cameras_1/`` directory because the capture rig
  is fixed.
- Lighting: each view has 7 ``rect_<VVV>_<L>_r5000.png`` files for light
  conditions 0..6. The canonical DTU eval uses ``L=3``; override via the
  ``light`` kwarg if a paper specifies otherwise.
- GT point clouds live in ``Points/stl/stl<SCAN:03d>_total.ply`` in the
  scanner frame (millimetres, typically; MVSNet converts to metres by
  dividing intrinsics but keeps PLY coords as-is). See
  ``DTU_POINT_SCALE`` below for the conversion.
- Sample GT ``point_cloud_gt`` is the per-scan PLY; ``depth_gt`` is left
  as ``None`` — the chamfer path against the GT scan is what VGGT and
  MASt3R report.

Standard 22-scan MVS test set (Galliani et al., adopted by MVSNet and
every learning-based MVS paper since):
    1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49,
    62, 75, 77, 110, 114, 118
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
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
    load_ply_xyz,
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = [
    "DTU_MVS_TEST_SCANS",
    "DTUDataset",
    "load_dtu_cam",
]

# The 22-scan MVS test set adopted by MVSNet / MASt3R / VGGT. See module
# docstring for provenance.
DTU_MVS_TEST_SCANS: tuple[int, ...] = (
    1,
    4,
    9,
    10,
    11,
    12,
    13,
    15,
    23,
    24,
    29,
    32,
    33,
    34,
    48,
    49,
    62,
    75,
    77,
    110,
    114,
    118,
)

# DTU's Points/stl/*.ply are in millimetres (the scanner's native unit);
# MVSNet's cam files express translation in millimetres to match. Chamfer
# numbers reported by MASt3R / VGGT are also in millimetres (Table 2 shows
# 0.382 "overall" which only makes sense as mm). Loader keeps both the
# point cloud and extrinsic translations in millimetres so the chamfer
# metric operates in one consistent unit; callers that want metres can
# divide by ``DTU_POINT_SCALE``.
DTU_POINT_SCALE: float = 1.0  # pass-through; units stay mm end-to-end


@register_dataset("dtu")
class DTUDataset(Dataset):
    """DTU MVS dataset loader (MVSNet-repacked layout).

    Each sample is one scan with ``views_per_sample`` consecutive views
    and its GT point cloud.

    Parameters
    ----------
    root
        Dataset root. Falls back to ``$DTU_ROOT``.
    split
        ``"test"`` (default, 22 scans per :data:`DTU_MVS_TEST_SCANS`) or
        ``"custom"`` when ``scans`` is given.
    scans
        Explicit list of scan IDs (integers). Takes precedence over
        ``split``. Use to evaluate a single scan for dev or to pin the
        exact subset a paper reports on.
    views_per_sample
        Views grouped into each sample. VGGT runs the paper-match at
        8 views; pass ``views_per_sample=49`` (the full ring) only on
        enough VRAM to hold it.
    light
        Lighting index 0..6 to use. Canonical MVS eval is ``3``.
    max_gt_points
        If set, deterministically subsample the GT point cloud to this
        many points. DTU scans are ~1-10M points — a 200k subsample
        keeps chamfer tractable without changing the value meaningfully.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        scans: list[int] | None = None,
        views_per_sample: int = 8,
        light: int = 3,
        max_gt_points: int | None = 200_000,
        gt_subsample_seed: int = 0,
    ) -> None:
        root_path = Path(root) if root else env_path("DTU_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "DTU not found. Set --data-root or $DTU_ROOT to the MVSNet-repacked "
                "DTU directory (Cameras_1/, Rectified/scan*_train/, Points/stl/stl*_total.ply). "
                "Public (no ToS) download: https://roboimagedata.compute.dtu.dk/?page_id=36 "
                "plus the MVSNet preprocessing: https://github.com/YoYo000/MVSNet."
            )
        if not 0 <= light <= 6:
            raise ValueError(f"light must be in 0..6; got {light}")
        if views_per_sample < 1:
            raise ValueError(f"views_per_sample must be >= 1; got {views_per_sample}")

        if scans is not None:
            scan_ids = [int(s) for s in scans]
            split_name = "custom"
        elif split == "test":
            scan_ids = list(DTU_MVS_TEST_SCANS)
            split_name = "test"
        else:
            raise ValueError(f"DTU split '{split}' unsupported; use 'test' or pass scans=[...]")

        self.root = root_path
        self.split = split_name
        self.scan_ids = scan_ids
        self.views_per_sample = int(views_per_sample)
        self.light = int(light)
        self.max_gt_points = max_gt_points
        self.gt_subsample_seed = int(gt_subsample_seed)

        cameras_dir = self.root / "Cameras_1"
        if not cameras_dir.exists():
            raise DatasetNotAvailable(
                f"Expected {cameras_dir}; not found. This loader targets the "
                "MVSNet-repacked layout — see module docstring for the expected tree."
            )
        self.cameras_dir = cameras_dir

        manifest_path = (
            self.root
            / ".plumbline_manifest"
            / f"dtu_{split_name}_vps{self.views_per_sample}_L{self.light}_n{len(scan_ids)}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan(scan_ids))
            save_manifest(manifest_path, records)
        self._records = records

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self, scan_ids: list[int]) -> Iterator[dict[str, Any]]:
        rectified = self.root / "Rectified"
        if not rectified.exists():
            raise DatasetNotAvailable(f"Expected {rectified}; not found.")
        for scan_id in scan_ids:
            scan_dir = rectified / f"scan{scan_id}_train"
            if not scan_dir.exists():
                # Skip missing scans silently when custom list is provided;
                # for the fixed test split, scan loss is a real error.
                if self.split == "test":
                    raise DatasetNotAvailable(
                        f"Required test-split scan directory missing: {scan_dir}"
                    )
                continue
            # DTU filenames are rect_<VVV>_<L>_r5000.png with VVV one-indexed
            # 001..049. We enumerate all available views at the chosen light.
            img_paths = sorted(scan_dir.glob(f"rect_*_{self.light}_r5000.png"))
            if not img_paths:
                continue

            # Per-view cam file: 0-indexed in Cameras_1/.
            view_indices = [_view_index_from_filename(p.name) for p in img_paths]
            # Construct sliding windows over available views (0-indexed).
            ordered = sorted(zip(view_indices, img_paths, strict=True))

            gt_ply = self.root / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
            gt_rel = str(gt_ply.relative_to(self.root)) if gt_ply.exists() else None

            for i in range(0, len(ordered) - self.views_per_sample + 1):
                group = ordered[i : i + self.views_per_sample]
                first_view = group[0][0]
                yield {
                    "sample_id": f"scan{scan_id}/view{first_view:03d}_v{self.views_per_sample}",
                    "scan_id": scan_id,
                    "view_indices": [v for v, _ in group],
                    "image_paths": [str(p.relative_to(self.root)) for _, p in group],
                    "gt_ply": gt_rel,
                }

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        imgs = [read_rgb_uint8(self.root / p) for p in rec["image_paths"]]
        # DTU rectified images are uniform-size across views per scan, so a
        # straight stack works — no padding dance needed (unlike ETH3D).
        images = np.stack(imgs, axis=0)
        assert_valid_image(images, name=f"dtu/{rec['sample_id']}/image")

        Ks: list[NDArray[np.float64]] = []
        cam_from_world: list[NDArray[np.float64]] = []
        for view_idx in rec["view_indices"]:
            cam_path = self.cameras_dir / f"{view_idx:08d}_cam.txt"
            K, E_cw = load_dtu_cam(cam_path)
            Ks.append(K)
            cam_from_world.append(E_cw)
        intrinsics = np.stack(Ks).astype(np.float32)
        world_from_camera = np.stack([invert_pose(p) for p in cam_from_world])
        extrinsics = rebase_to_first_camera(world_from_camera).astype(np.float32)

        assert_valid_intrinsics(intrinsics, name=f"dtu/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(extrinsics, name=f"dtu/{rec['sample_id']}/extrinsics")

        pcd: NDArray[np.float32] | None = None
        if rec.get("gt_ply"):
            pcd_path = self.root / rec["gt_ply"]
            if pcd_path.exists():
                pcd = load_ply_xyz(pcd_path)
                if self.max_gt_points is not None and pcd.shape[0] > self.max_gt_points:
                    rng = np.random.default_rng(self.gt_subsample_seed)
                    idx = rng.choice(pcd.shape[0], size=self.max_gt_points, replace=False)
                    pcd = pcd[idx]

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            point_cloud_gt=pcd,
            metadata={
                "scan_id": rec["scan_id"],
                "view_indices": rec["view_indices"],
                "light": self.light,
                "split": self.split,
                "units": "mm",
            },
        )


# ---------------------------------------------------------------------------
# Cam-file parser
# ---------------------------------------------------------------------------


def load_dtu_cam(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse an MVSNet-style DTU ``_cam.txt`` file.

    Format::

        extrinsic
        e00 e01 e02 e03
        e10 e11 e12 e13
        e20 e21 e22 e23
        e30 e31 e32 e33

        intrinsic
        k00 k01 k02
        k10 k11 k12
        k20 k21 k22

        DEPTH_MIN  DEPTH_INTERVAL  [DEPTH_NUM]  [DEPTH_MAX]

    Returns ``(K, E_cam_from_world)`` — plumbline's canonical convention
    inverts this to ``world_from_camera`` in the loader.
    """
    text = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(line.split())
    # Expected layout: "extrinsic" + 16 floats + "intrinsic" + 9 floats +
    # trailing depth-range values (we ignore them).
    try:
        i_ext = tokens.index("extrinsic")
        i_int = tokens.index("intrinsic", i_ext + 1)
    except ValueError as exc:
        raise ValueError(
            f"{path}: expected 'extrinsic' and 'intrinsic' markers in cam file"
        ) from exc
    ext_values = tokens[i_ext + 1 : i_int]
    if len(ext_values) < 16:
        raise ValueError(
            f"{path}: expected 16 extrinsic floats after 'extrinsic'; got {len(ext_values)}"
        )
    int_values = tokens[i_int + 1 : i_int + 10]
    if len(int_values) < 9:
        raise ValueError(
            f"{path}: expected 9 intrinsic floats after 'intrinsic'; got {len(int_values)}"
        )
    E = np.asarray([float(x) for x in ext_values[:16]], dtype=np.float64).reshape(4, 4)
    K = np.asarray([float(x) for x in int_values[:9]], dtype=np.float64).reshape(3, 3)
    return K, E


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _view_index_from_filename(name: str) -> int:
    """``'rect_007_3_r5000.png'`` -> 6 (0-indexed view for Cameras_1/).

    DTU's image filenames use a 1-indexed view number (001..049) while
    ``Cameras_1/`` names its cam files 0-indexed (00000000..00000048).
    """
    # Parse the three-digit field after 'rect_'.
    try:
        parts = name.split("_")
        view_1based = int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unexpected DTU image filename {name!r}") from exc
    return view_1based - 1
