"""ETH3D high-resolution multi-view loader.

The ETH3D high-res train split is the standard benchmark for high-resolution
multi-view stereo. Ground truth is a laser-scanned point cloud plus calibrated
cameras in the COLMAP ``.txt`` format.

Expected layout (point ``--data-root`` or ``$ETH3D_ROOT`` here)::

    <root>/<scene>/
      images/dslr_images_undistorted/<image_name>.JPG  # native (often 6K)
      dslr_calibration_undistorted/
        cameras.txt
        images.txt
        points3D.txt
      scan_clean/scan*.ply   # ground-truth laser scan (ETH3D ships multiple)
      # or the legacy flat form:
      # scan_clean.ply

Access: https://www.eth3d.net/high_res_multi_view (public; large downloads).
Fetch a single scene end-to-end with:

    curl -L --fail -O https://www.eth3d.net/data/<scene>_dslr_undistorted.7z
    curl -L --fail -O https://www.eth3d.net/data/<scene>_scan_clean.7z
    7z x -y <scene>_dslr_undistorted.7z
    7z x -y <scene>_scan_clean.7z

Conventions
-----------
- COLMAP poses are ``camera_from_world`` (qw, qx, qy, qz, tx, ty, tz). We
  convert to 4x4, then invert, then rebase to first camera.
- Intrinsics are per-camera-id (ETH3D groups images by rig). We carry the
  intrinsic matching each image's ``camera_id``.
- We *do not* parse ``points3D.txt`` (sparse SfM points). The dense GT lives
  in ``scan_clean.ply``; load it lazily through :meth:`_load_point_cloud_gt`.
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
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["ETH3DDataset", "parse_colmap_cameras", "parse_colmap_images", "quat_to_rot"]


@register_dataset("eth3d")
class ETH3DDataset(Dataset):
    """ETH3D high-res multi-view dataset loader.

    Parameters
    ----------
    root
        Dataset root: ``<root>/<scene>/...``.
    split
        ``"train"`` (default; public with GT) or ``"test"`` (no public GT).
    scenes
        Optional scene whitelist.
    views_per_sample
        Views grouped into each sample. Default 4 (a modest multi-view set).
    """

    split: str = "train"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "train",
        scenes: list[str] | None = None,
        views_per_sample: int = 4,
        max_gt_points: int | None = None,
        gt_subsample_seed: int = 0,
    ) -> None:
        root_path = Path(root) if root else env_path("ETH3D_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ETH3D not found. Set --data-root or $ETH3D_ROOT to a directory "
                "containing <scene>/images/*.JPG and "
                "<scene>/dslr_calibration_undistorted/{cameras,images}.txt. "
                "Download from https://www.eth3d.net/high_res_multi_view."
            )
        if split not in ("train",):
            raise ValueError(f"ETH3D split '{split}' unsupported; use 'train' (no public test GT).")
        self.root = root_path
        self.split = split
        self.views_per_sample = max(1, int(views_per_sample))

        manifest_path = (
            self.root / ".plumbline_manifest" / f"eth3d_{split}_vps{self.views_per_sample}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan(scenes))
            save_manifest(manifest_path, records)
        if scenes:
            records = [r for r in records if r["scene"] in scenes]
        self._records = records
        self.max_gt_points = max_gt_points
        self.gt_subsample_seed = int(gt_subsample_seed)

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self, scenes: list[str] | None) -> Iterator[dict[str, Any]]:
        scene_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if scenes is not None:
            wanted = set(scenes)
            scene_dirs = [p for p in scene_dirs if p.name in wanted]
        for scene_dir in scene_dirs:
            calib = scene_dir / "dslr_calibration_undistorted"
            if not (calib / "images.txt").exists():
                continue
            images_info = parse_colmap_images(calib / "images.txt")
            # Stable deterministic order (ascending image_id).
            ordered = sorted(images_info, key=lambda x: x["image_id"])
            ply_paths = _resolve_scan_clean_plys(scene_dir)
            for i in range(0, len(ordered) - self.views_per_sample + 1):
                group = ordered[i : i + self.views_per_sample]
                yield {
                    "sample_id": f"{scene_dir.name}/{group[0]['image_id']:06d}_v{self.views_per_sample}",
                    "scene": scene_dir.name,
                    "image_records": group,
                    "cameras_txt": str((calib / "cameras.txt").relative_to(self.root)),
                    "point_cloud_plys": [str(p.relative_to(self.root)) for p in ply_paths]
                    if ply_paths
                    else None,
                }

    # -- per sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        cameras = parse_colmap_cameras(self.root / rec["cameras_txt"])
        cam_for_image: dict[int, NDArray[np.float32]] = {}

        images: list[NDArray[np.uint8]] = []
        Ks: list[NDArray[np.float32]] = []
        poses_cw: list[NDArray[np.float64]] = []

        for ir in rec["image_records"]:
            path = self.root / rec["scene"] / "images" / ir["name"]
            img = read_rgb_uint8(path)
            images.append(img)
            K = cameras[ir["camera_id"]]
            Ks.append(K)
            cam_for_image[ir["image_id"]] = K

            q = np.array([ir["qw"], ir["qx"], ir["qy"], ir["qz"]], dtype=np.float64)
            t = np.array([ir["tx"], ir["ty"], ir["tz"]], dtype=np.float64)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = quat_to_rot(q)
            pose[:3, 3] = t
            poses_cw.append(pose)  # camera_from_world per COLMAP

        # Images may differ in size; pack into the largest canvas with zero
        # padding for now. Adapters that care should use the native-size
        # per-view from metadata; most models resize internally.
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        padded = np.zeros((len(images), max_h, max_w, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            h, w, _ = img.shape
            padded[i, :h, :w] = img

        assert_valid_image(padded, name=f"eth3d/{rec['sample_id']}/image")

        intrinsics = np.stack(Ks).astype(np.float32)
        assert_valid_intrinsics(intrinsics, name=f"eth3d/{rec['sample_id']}/intrinsics")

        world_from_camera = np.stack([invert_pose(p) for p in poses_cw])
        extrinsics = rebase_to_first_camera(world_from_camera).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name=f"eth3d/{rec['sample_id']}/extrinsics")

        pcd = None
        # Newer manifests store a list of ply paths (ETH3D ships scan_clean
        # as multiple files); older manifests stored a single path under
        # "point_cloud_ply". Accept both so caches don't need a rebuild.
        ply_rels = rec.get("point_cloud_plys")
        if ply_rels is None and rec.get("point_cloud_ply"):
            ply_rels = [rec["point_cloud_ply"]]
        if ply_rels:
            chunks: list[NDArray[np.float32]] = []
            for rel in ply_rels:
                p = self.root / rel
                if p.exists():
                    chunks.append(_load_ply_xyz(p))
            if chunks:
                pcd = np.concatenate(chunks, axis=0)
                # ETH3D scan_clean is ~38M points/scene; chamfer on that is
                # minutes per sample. Opt-in subsample makes smoke evals
                # practical. Deterministic seed + per-sample stable choice
                # so the same sample always gets the same subset.
                if self.max_gt_points is not None and pcd.shape[0] > self.max_gt_points:
                    rng = np.random.default_rng(self.gt_subsample_seed)
                    idx = rng.choice(pcd.shape[0], size=self.max_gt_points, replace=False)
                    pcd = pcd[idx]

        return Sample(
            sample_id=rec["sample_id"],
            images=padded,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            point_cloud_gt=pcd,
            metadata={
                "scene": rec["scene"],
                "native_sizes": [(img.shape[0], img.shape[1]) for img in images],
                "split": self.split,
            },
        )


# ---------------------------------------------------------------------------
# COLMAP parsers
# ---------------------------------------------------------------------------


def parse_colmap_cameras(path: Path) -> dict[int, NDArray[np.float32]]:
    """Parse a COLMAP ``cameras.txt`` into ``{camera_id: K}`` (float32 3x3).

    ETH3D uses ``PINHOLE`` cameras (fx, fy, cx, cy) after undistortion. Other
    COLMAP camera models are not handled here; extend when needed.
    """
    cameras: dict[int, NDArray[np.float32]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            params = [float(x) for x in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                f_, cx, cy = params[:3]
                fx = fy = f_
            else:
                raise ValueError(f"ETH3D loader only handles PINHOLE; got model '{model}'")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            cameras[camera_id] = K
    return cameras


def parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP ``images.txt`` skipping the 2D-point lines.

    Yields one dict per image with keys ``image_id``, ``qw..qz``, ``tx..tz``,
    ``camera_id``, ``name``.
    """
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rec = {
                "image_id": int(parts[0]),
                "qw": float(parts[1]),
                "qx": float(parts[2]),
                "qy": float(parts[3]),
                "qz": float(parts[4]),
                "tx": float(parts[5]),
                "ty": float(parts[6]),
                "tz": float(parts[7]),
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
            out.append(rec)
            # Skip the 2D points line that follows.
            f.readline()
    return out


def quat_to_rot(q: NDArray[Any]) -> NDArray[np.float64]:
    """COLMAP quaternion ``(qw, qx, qy, qz)`` → 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _resolve_scan_clean_plys(scene_dir: Path) -> list[Path]:
    """Find the ground-truth laser scan ply files for a scene.

    ETH3D archives ship ``scan_clean`` as a directory with one or more
    ``scan*.ply`` files; older, manually-prepared mirrors sometimes place
    a single ``scan_clean.ply`` at the scene root. Return a sorted list of
    paths, or an empty list if neither is present.
    """
    direct = scene_dir / "scan_clean.ply"
    if direct.exists():
        return [direct]
    subdir = scene_dir / "scan_clean"
    if subdir.is_dir():
        return sorted(subdir.glob("scan*.ply"))
    return []


_PLY_PROP_BYTES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


def _load_ply_xyz(path: Path) -> NDArray[np.float32]:
    """Minimal PLY parser: returns ``(N, 3)`` float32 XYZ only.

    Supports ``ascii`` and ``binary_little_endian`` with ``float`` XYZ as the
    first three vertex properties. Computes the vertex stride from the
    header so files with multiple ``element`` blocks (e.g. ETH3D's
    ``scan_clean`` PLYs that append a trailing ``element camera`` with
    sensor metadata) don't mislead the reshape.
    """
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line.startswith("end_header"):
                break
        fmt = next(
            (ln.split()[1] for ln in header_lines if ln.startswith("format")),
            "ascii",
        )
        # Vertex element + its property widths. Ignore any later elements.
        vcount = 0
        vertex_props: list[str] = []
        in_vertex = False
        for ln in header_lines:
            if ln.startswith("element "):
                parts = ln.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vcount = int(parts[2])
            elif in_vertex and ln.startswith("property "):
                parts = ln.split()
                # "property <type> <name>" (skip list properties — not used on
                # ETH3D scan_clean).
                if parts[1] == "list":
                    raise NotImplementedError("list properties unsupported")
                vertex_props.append(parts[1])
        payload = f.read()

    if fmt.startswith("binary_little_endian"):
        vertex_stride = sum(_PLY_PROP_BYTES[p] for p in vertex_props)
        vertex_bytes = vcount * vertex_stride
        buf = np.frombuffer(payload[:vertex_bytes], dtype=np.uint8).reshape(
            vcount, vertex_stride
        )
        xyz = np.frombuffer(buf[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
        return np.ascontiguousarray(xyz)

    xyz = np.empty((vcount, 3), dtype=np.float32)
    for i, line in enumerate(payload.decode("ascii").splitlines()[:vcount]):
        parts = line.split()
        xyz[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return xyz
