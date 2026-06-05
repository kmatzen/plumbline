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

Download (public, no ToS):
Two archives together give the full 22-scan paper-match setup:

1. **MVSNet preprocessed test set** (~554 MB). Provides all 22 test
   scans' images + cameras + per-scan pair.txt in the ``dtu/scanN/
   {cams,images}/`` layout (no GT point clouds). Hosted on Google
   Drive; fetch with `gdown`::

       pip install gdown
       gdown "135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_" -O dtu_test.zip
       unzip dtu_test.zip -d $DTU_ROOT

2. **DTU GT point clouds** (~6.97 GB). The ``Points/`` subtree with
   ``stl*_total.ply`` per scan is what chamfer eval compares against::

       curl -L -o Points.zip \\
           https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip
       unzip Points.zip -d $DTU_ROOT

Do NOT confuse the above with DTU's ``SampleSet.zip`` (also ~6.9 GB
on the same server) — that is a **format-demo with scans 1 & 6 only**,
not the eval set. The plumbline v0.1 gate (VGGT Table 2 chamfer=0.382
on 22-scan test) requires both of the two archives above.

Also available: MVSNet's full preprocessed training/validation split
(~115 GB) at https://roboimagedata.compute.dtu.dk/?page_id=36 +
https://github.com/YoYo000/MVSNet — overkill for eval-only needs.

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
    "render_pv_depth_zbuffer",
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
    with_per_view_gt
        When ``True``, render per-view GT depth + validity masks by
        z-buffering the full laser PLY through each view's GT camera.
        This unlocks the **per-view-masked chamfer** protocol the VGGT
        paper follows (CUT3R/MASt3R lineage, see
        ``/tmp/cut3r/eval/mv_recon/data.py::DTU._get_views``). Adds
        ``Sample.depth_gt`` ``(N,H,W) float32 mm`` and
        ``Sample.depth_valid`` ``(N,H,W) bool``. Rendered depth is
        cached per scan as ``<root>/.plumbline_manifest/dtu_pv_depth_
        scan{N}.npz`` so the cost is one-time. Off by default to
        preserve the legacy scene-merged path.
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
        points_root: Path | str | None = None,
        with_per_view_gt: bool = False,
        pv_splat_radius: int = 1,
    ) -> None:
        root_path = Path(root) if root else env_path("DTU_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "DTU not found. Set --data-root or $DTU_ROOT to the MVSNet-repacked "
                "DTU directory. Two layouts are auto-detected: (1) the training-format "
                "'Cameras_1/ + Rectified/scanN_train/' repack, and (2) the test-format "
                "'dtu/scanN/{cams,images}/' repack that ships as dtu_test.zip "
                "(MVSNet Google Drive). GT point clouds live in Points/stl/ or are "
                "overridable via points_root. Public download: "
                "https://roboimagedata.compute.dtu.dk/?page_id=36 ; "
                "https://github.com/YoYo000/MVSNet."
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
        self.with_per_view_gt = bool(with_per_view_gt)
        self.pv_splat_radius = int(pv_splat_radius)

        # Auto-detect layout. Prefer the training-format repack when both are
        # present (it has shared Cameras_1 that's more authoritative).
        layout, scans_root, cameras_dir = _detect_dtu_layout(root_path)
        if layout is None:
            raise DatasetNotAvailable(
                f"Expected a DTU layout under {root_path}. Either "
                "Cameras_1/ (training-format) or <root>/[dtu/]scanN/cams/ "
                "(test-format) must exist. See module docstring."
            )
        self.layout = layout
        self._scans_root = scans_root
        self.cameras_dir = cameras_dir  # None when layout == "per_scan"

        # GT point-cloud source: explicit override wins; else search
        # <root>/Points/stl/, <root>/SampleSet/MVS Data/Points/stl/.
        if points_root is not None:
            self.points_root: Path | None = Path(points_root)
        else:
            self.points_root = _detect_points_root(root_path)

        manifest_path = (
            self.root
            / ".plumbline_manifest"
            / f"dtu_{split_name}_{layout}_vps{self.views_per_sample}_L{self.light}_n{len(scan_ids)}.jsonl"
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
        if self.layout == "training":
            yield from self._scan_training(scan_ids)
        else:
            yield from self._scan_per_scan(scan_ids)

    def _scan_training(self, scan_ids: list[int]) -> Iterator[dict[str, Any]]:
        """Training-format repack: Rectified/scanN_train/rect_VVV_L_r5000.png."""
        assert self._scans_root is not None
        for scan_id in scan_ids:
            scan_dir = self._scans_root / f"scan{scan_id}_train"
            if not scan_dir.exists():
                if self.split == "test":
                    raise DatasetNotAvailable(
                        f"Required test-split scan directory missing: {scan_dir}"
                    )
                continue
            img_paths = sorted(scan_dir.glob(f"rect_*_{self.light}_r5000.png"))
            if not img_paths:
                continue
            view_indices = [_view_index_from_filename(p.name) for p in img_paths]
            ordered = sorted(zip(view_indices, img_paths, strict=True))
            gt_rel = self._locate_gt_ply(scan_id)
            for i in range(0, len(ordered) - self.views_per_sample + 1):
                group = ordered[i : i + self.views_per_sample]
                first_view = group[0][0]
                yield {
                    "sample_id": f"scan{scan_id}/view{first_view:03d}_v{self.views_per_sample}",
                    "scan_id": scan_id,
                    "view_indices": [v for v, _ in group],
                    "image_paths": [str(p.relative_to(self.root)) for _, p in group],
                    "cam_paths": None,  # shared Cameras_1/ lookup happens in _load_sample
                    "gt_ply": gt_rel,
                }

    def _scan_per_scan(self, scan_ids: list[int]) -> Iterator[dict[str, Any]]:
        """Test-format repack: <scans_root>/scanN/{images,cams}/ per scan.

        Filenames are 0-indexed (``00000000.jpg``, ``00000000_cam.txt``);
        light is not encoded in the test archive so ``self.light`` is
        ignored in this layout.
        """
        assert self._scans_root is not None
        for scan_id in scan_ids:
            scan_dir = self._scans_root / f"scan{scan_id}"
            if not scan_dir.exists():
                if self.split == "test":
                    raise DatasetNotAvailable(
                        f"Required test-split scan directory missing: {scan_dir}"
                    )
                continue
            img_dir = scan_dir / "images"
            cam_dir = scan_dir / "cams"
            if not (img_dir.exists() and cam_dir.exists()):
                continue
            img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if not img_paths:
                continue
            # Filenames encode 0-indexed view; pair with matching cam file.
            triples: list[tuple[int, Path, Path]] = []
            for p in img_paths:
                view_idx = int(p.stem)
                cam_path = cam_dir / f"{view_idx:08d}_cam.txt"
                if cam_path.exists():
                    triples.append((view_idx, p, cam_path))
            ordered = sorted(triples)
            gt_rel = self._locate_gt_ply(scan_id)
            for i in range(0, len(ordered) - self.views_per_sample + 1):
                group = ordered[i : i + self.views_per_sample]
                first_view = group[0][0]
                yield {
                    "sample_id": f"scan{scan_id}/view{first_view:03d}_v{self.views_per_sample}",
                    "scan_id": scan_id,
                    "view_indices": [v for v, _, _ in group],
                    "image_paths": [str(p.relative_to(self.root)) for _, p, _ in group],
                    "cam_paths": [str(c.relative_to(self.root)) for _, _, c in group],
                    "gt_ply": gt_rel,
                }

    def _locate_gt_ply(self, scan_id: int) -> str | None:
        """Return a relative path to stl<scan_id>_total.ply if present anywhere
        under the configured points_root, else None."""
        if self.points_root is None or not self.points_root.exists():
            return None
        # Try a few standard names; original DTU sometimes has trailing spaces.
        candidates = [
            self.points_root / "stl" / f"stl{scan_id:03d}_total.ply",
            self.points_root / f"stl{scan_id:03d}_total.ply",
        ]
        for c in candidates:
            if c.exists():
                # Return relative to self.root when possible so manifests stay
                # portable across machines.
                try:
                    return str(c.relative_to(self.root))
                except ValueError:
                    return str(c)
        return None

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        imgs = [read_rgb_uint8(self.root / p) for p in rec["image_paths"]]
        # DTU rectified images are uniform-size across views per scan, so a
        # straight stack works — no padding dance needed (unlike ETH3D).
        images = np.stack(imgs, axis=0)
        assert_valid_image(images, name=f"dtu/{rec['sample_id']}/image")

        Ks: list[NDArray[np.float64]] = []
        cam_from_world: list[NDArray[np.float64]] = []
        cam_rel_paths = rec.get("cam_paths")
        for i, view_idx in enumerate(rec["view_indices"]):
            if cam_rel_paths is not None:
                cam_path = self.root / cam_rel_paths[i]
            else:
                # training-format: shared Cameras_1/ lookup by 0-indexed view
                assert self.cameras_dir is not None
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
        pcd_full: NDArray[np.float32] | None = None
        if rec.get("gt_ply"):
            pcd_path = self.root / rec["gt_ply"]
            if pcd_path.exists():
                pcd_full = load_ply_xyz(pcd_path)
                if self.max_gt_points is not None and pcd_full.shape[0] > self.max_gt_points:
                    rng = np.random.default_rng(self.gt_subsample_seed)
                    idx = rng.choice(pcd_full.shape[0], size=self.max_gt_points, replace=False)
                    pcd = pcd_full[idx]
                else:
                    pcd = pcd_full

        depth_gt: NDArray[np.float32] | None = None
        depth_valid: NDArray[np.bool_] | None = None
        if self.with_per_view_gt and pcd_full is not None:
            # Per-scan cache. Render once at native (1200, 1600) and slice
            # to whatever views this sample needs. The render is purely a
            # function of (full PLY, all-49 cam files, image dims), so the
            # cache key is the scan id alone.
            H, W = images.shape[1], images.shape[2]
            cache_path = (
                self.root
                / ".plumbline_manifest"
                / f"dtu_pv_depth_scan{rec['scan_id']}_HxW{H}x{W}_r{self.pv_splat_radius}.npz"
            )
            depths_per_scan_view, valids_per_scan_view = self._load_or_render_pv_depth(
                cache_path,
                rec["scan_id"],
                pcd_full,
                H,
                W,
                splat_radius=self.pv_splat_radius,
            )
            view_indices = rec["view_indices"]
            depth_gt = depths_per_scan_view[view_indices].astype(np.float32, copy=False)
            depth_valid = valids_per_scan_view[view_indices].astype(np.bool_, copy=False)

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            point_cloud_gt=pcd,
            metadata={
                "scan_id": rec["scan_id"],
                "view_indices": rec["view_indices"],
                "light": self.light,
                "split": self.split,
                "units": "mm",
            },
        )

    def _load_or_render_pv_depth(
        self,
        cache_path: Path,
        scan_id: int,
        xyz: NDArray[np.float32],
        H: int,
        W: int,
        *,
        splat_radius: int = 1,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Render z-buffered per-view depth for every view of this scan.

        Returns ``(depths, valids)`` shaped ``(V_total, H, W)`` where
        ``V_total`` is the count of cam files belonging to this scan
        (49 for the canonical DTU rig). Caches to ``cache_path``.
        """
        if cache_path.exists():
            with np.load(cache_path) as f:
                return f["depths"].astype(np.float32), f["valids"].astype(np.bool_)

        # Discover all cam files for this scan in cam-file order.
        if self.layout == "training":
            assert self.cameras_dir is not None
            cam_paths = sorted(self.cameras_dir.glob("*_cam.txt"))
        else:
            scan_dir = self._scans_root / f"scan{scan_id}"  # type: ignore[operator]
            cam_paths = sorted((scan_dir / "cams").glob("*_cam.txt"))
        if not cam_paths:
            raise FileNotFoundError(
                f"no cam files found while rendering per-view GT for scan{scan_id}"
            )
        depths = np.zeros((len(cam_paths), H, W), dtype=np.float32)
        valids = np.zeros((len(cam_paths), H, W), dtype=np.bool_)
        for i, cp in enumerate(cam_paths):
            K_v, E_cw_v = load_dtu_cam(cp)
            d, v = render_pv_depth_zbuffer(xyz, K_v, E_cw_v, H, W, splat_radius=splat_radius)
            depths[i] = d
            valids[i] = v
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, depths=depths, valids=valids)
        return depths, valids


# ---------------------------------------------------------------------------
# Per-view GT rendering
# ---------------------------------------------------------------------------


def render_pv_depth_zbuffer(
    xyz: NDArray[np.floating],
    K: NDArray[np.floating],
    cam_from_world: NDArray[np.floating],
    height: int,
    width: int,
    *,
    splat_radius: int = 1,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Z-buffer-render a point cloud through a pinhole camera.

    Each PLY point splats to a square neighbourhood of size
    ``(2*splat_radius+1)^2`` and per-pixel we keep the nearest depth.
    Output ``depth`` is float32 with zeros where no point landed;
    ``valid`` is the corresponding boolean mask. Used by
    :class:`DTUDataset` to derive per-view depth + visibility from
    DTU's scene-level laser PLY (matches the canonical
    "render-the-laser-surface" provenance of MVSNet's preprocessed
    ``depths/<view>.npy`` + ``binary_masks/<view>.png``).

    Notes
    -----
    DTU PLYs ship with ~3 M points per scan; at native 1200x1600 res
    that gives ~50% pixel coverage with ``splat_radius=0`` (single
    pixel) and ~95%+ with ``splat_radius=1`` (3x3 disk). Default 1
    matches the density of MVSNet's Poisson-mesh-rendered depth maps
    that the CUT3R/MASt3R/VGGT-family DTU eval consumes.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(cam_from_world, dtype=np.float64)
    R = E[:3, :3]
    t = E[:3, 3]
    p_cam = xyz @ R.T + t
    z = p_cam[:, 2]
    front = z > 0.0
    if not front.any():
        return np.zeros((height, width), dtype=np.float32), np.zeros(
            (height, width), dtype=np.bool_
        )
    p_cam = p_cam[front]
    z = z[front]
    u = K[0, 0] * p_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / z + K[1, 2]
    ui_c = np.floor(u).astype(np.int64)
    vi_c = np.floor(v).astype(np.int64)
    if splat_radius > 0:
        # Tile each (vi_c, ui_c) over a (2r+1)^2 square, expanding
        # the index arrays by that factor in lock-step with z.
        offsets = np.arange(-splat_radius, splat_radius + 1, dtype=np.int64)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy_flat = dy.reshape(-1)
        dx_flat = dx.reshape(-1)
        ui_all = (ui_c[:, None] + dx_flat[None, :]).reshape(-1)
        vi_all = (vi_c[:, None] + dy_flat[None, :]).reshape(-1)
        z_all = np.broadcast_to(z[:, None], (z.shape[0], dx_flat.shape[0])).reshape(-1)
        ui_c, vi_c, z = ui_all, vi_all, z_all
    in_image = (ui_c >= 0) & (ui_c < width) & (vi_c >= 0) & (vi_c < height)
    ui_c = ui_c[in_image]
    vi_c = vi_c[in_image]
    z = z[in_image]
    if ui_c.size == 0:
        return np.zeros((height, width), dtype=np.float32), np.zeros(
            (height, width), dtype=np.bool_
        )
    flat = vi_c * width + ui_c
    # Per-pixel min-z via ``np.minimum.at`` — atomic min reduction, one
    # pass over all (flat, z) pairs. Faster than sort+unique when the
    # splat radius blows ``len(z)`` up to tens of millions.
    depth_buf = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(depth_buf, flat, z)
    valid = depth_buf < np.inf
    depth = np.zeros(height * width, dtype=np.float32)
    depth[valid] = depth_buf[valid].astype(np.float32)
    return depth.reshape(height, width), valid.reshape(height, width)


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


def _detect_dtu_layout(
    root: Path,
) -> tuple[str | None, Path | None, Path | None]:
    """Auto-detect DTU repack format.

    Returns ``(layout, scans_root, cameras_dir)`` where:
    - ``layout`` is ``"training"``, ``"per_scan"``, or ``None`` if neither
      matches.
    - ``scans_root`` is the directory that contains per-scan subdirs.
    - ``cameras_dir`` is the shared ``Cameras_1/`` (training layout only).

    Preference order: training layout first (more authoritative — shared
    Cameras_1 reflects the calibrated rig), then per_scan.
    """
    cameras_dir = root / "Cameras_1"
    rectified = root / "Rectified"
    if cameras_dir.exists() and rectified.exists():
        return "training", rectified, cameras_dir
    # Test-format repack may live either directly under root or under
    # "dtu/"  (matches the MVSNet Google Drive archive we distribute).
    for candidate in (root, root / "dtu"):
        if (candidate / "scan1").is_dir() and (candidate / "scan1" / "cams").is_dir():
            return "per_scan", candidate, None
    return None, None, None


def _detect_points_root(root: Path) -> Path | None:
    """Find the directory that holds DTU GT point clouds.

    Tries several known layouts in order:

    1. ``<root>/Points/``
    2. ``<root>/SampleSet/MVS Data/Points/`` (original DTU archive)
    3. ``<root>/Points/stl/``  (already-drilled-in variant)

    Returns ``None`` when nothing looks right; the loader then yields
    samples with ``point_cloud_gt=None`` and chamfer silently skips.
    """
    candidates = [
        root / "Points",
        root / "SampleSet" / "MVS Data" / "Points",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


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
