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
    load_ply_xyz,
    read_rgb_uint8,
    save_manifest,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = [
    "ETH3DDataset",
    "parse_colmap_cameras",
    "parse_colmap_images",
    "parse_scan_alignment_mlp",
    "quat_to_rot",
]


def _resize_rgb_uint8(image: NDArray[np.uint8], *, height: int, width: int) -> NDArray[np.uint8]:
    """Down/up-sample an RGB uint8 image to ``(height, width)`` with area filter."""
    from PIL import Image

    h, w, _ = image.shape
    if h == height and w == width:
        return image
    pil = Image.fromarray(image)
    resample = Image.Resampling.BOX if (height < h or width < w) else Image.Resampling.BILINEAR
    return np.asarray(pil.resize((width, height), resample=resample), dtype=np.uint8)


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
        with_per_view_gt: bool = False,
        pv_splat_radius: int = 1,
        pv_render_max_dim: int | None = 2048,
        resize_images_to_pv_render: bool = False,
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

        # Manifest caches the full on-disk scan (all scenes under root), and
        # the `scenes` whitelist is applied after load. An earlier revision
        # keyed the cache only on split+vps but saved a scene-filtered scan,
        # so a prior single-scene run left a cache that silently hid other
        # scenes from later multi-scene calls. Filename bumped to _v2 to
        # invalidate those stale caches on upgrade.
        #
        # D10b (2026-05-27): even with the "scan(None)" behavior, the cache
        # could go stale if new scene dirs are added to disk *after* the
        # manifest is written — e.g. a 3-scene staging followed by a
        # 10-scene top-up. We now compare the on-disk scene-dir set to the
        # manifest's scene set; if disk has any scene that the manifest
        # doesn't, re-scan + re-save. (Stale entries — manifest scenes
        # that no longer exist on disk — are tolerated; they yield zero
        # samples and don't otherwise affect downstream code.)
        manifest_path = (
            self.root / ".plumbline_manifest" / f"eth3d_{split}_vps{self.views_per_sample}_v2.jsonl"
        )
        on_disk_scenes = {
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / "dslr_calibration_undistorted" / "images.txt").exists()
        }
        records: list[dict[str, Any]] = []
        if manifest_path.exists():
            records = load_manifest(manifest_path)
            cached_scenes = {r["scene"] for r in records}
            missing_on_disk = on_disk_scenes - cached_scenes
            if missing_on_disk:
                # New scenes since the manifest was last written. Re-scan
                # the full set so the cache reflects current disk state.
                # (We don't try to incrementally merge — full re-scan is
                # cheap relative to inference.)
                records = list(self._scan(None))
                save_manifest(manifest_path, records)
        else:
            records = list(self._scan(None))
            save_manifest(manifest_path, records)
        if scenes:
            records = [r for r in records if r["scene"] in scenes]
        self._records = records
        self.max_gt_points = max_gt_points
        self.gt_subsample_seed = int(gt_subsample_seed)
        # Per-scene GT point cloud cache. Populated lazily on first sample
        # of a scene; subsequent samples reuse the same array. Without this,
        # _load_sample re-loads + re-parses the multi-PLY GT (~34M pts for
        # ETH3D courtyard / facade) on *every* sample — 137 samples × two
        # 200 MB PLY files = 45 GB of I/O + a Python-side heap-fragmentation
        # pattern that OOM-killed D4's scene-aggregation on 2026-04-24.
        # Keyed by (scene_name, tuple-of-relative-paths) so a scene that
        # somehow appears under two GT sources within one run still works.
        self._gt_cache: dict[tuple, NDArray[np.float32]] = {}
        self.with_per_view_gt = bool(with_per_view_gt)
        self.pv_splat_radius = int(pv_splat_radius)
        # Cap rendering resolution; ETH3D native is 6048x4032 which makes
        # both rendering time (per-view ~30 sec at native) and on-disk cache
        # (3.7 GB / scene) painful. 2048 max-dim keeps per-view cost ~3 sec
        # and per-scene cache ~400 MB while staying well above any typical
        # pred resolution (518 for VGGT).
        self.pv_render_max_dim = (
            int(pv_render_max_dim) if pv_render_max_dim is not None else None
        )
        # When True, resize each RGB view to the per-view GT render resolution
        # (``pv_render_max_dim`` cap) so mono-depth metrics are pixel-aligned.
        # VGGT/π³ chamfer runs keep False — they need native images for the
        # adapter while GT stays at the capped render res for masking.
        self.resize_images_to_pv_render = bool(resize_images_to_pv_render)
        # Per-scene rendered (depth, valid, image_id_to_idx) cache.
        self._pv_depth_cache: dict[str, dict[str, Any]] = {}

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

        depth_gt: NDArray[np.float32] | None = None
        depth_valid: NDArray[np.bool_] | None = None
        gt_sizes: list[tuple[int, int]] | None = None
        native_sizes = [(img.shape[0], img.shape[1]) for img in images]

        pcd = None
        # Newer manifests store a list of ply paths (ETH3D ships scan_clean
        # as multiple files); older manifests stored a single path under
        # "point_cloud_ply". Accept both so caches don't need a rebuild.
        ply_rels = rec.get("point_cloud_plys")
        if ply_rels is None and rec.get("point_cloud_ply"):
            ply_rels = [rec["point_cloud_ply"]]
        if ply_rels:
            cache_key = (rec["scene"], tuple(ply_rels))
            pcd = self._gt_cache.get(cache_key)
            if pcd is None:
                # Apply per-scan MLMatrix44 transforms from
                # ``scan_alignment.mlp`` if present, so all scan{N}.ply
                # files come out in the COLMAP / DSLR world frame
                # before concatenation. Without this, scans live in
                # individual scanner frames whose pairwise rotation
                # is up to ~14° (e.g. courtyard scan1) — concatenated
                # GT is then rotationally scrambled relative to pred,
                # which inflates Comp by 3–4× (D4 regression).
                scan_alignment: dict[str, NDArray[np.float64]] = {}
                if ply_rels:
                    mlp_dir = (self.root / ply_rels[0]).parent
                    mlp_path = mlp_dir / "scan_alignment.mlp"
                    if mlp_path.exists():
                        scan_alignment = parse_scan_alignment_mlp(mlp_path)
                chunks: list[NDArray[np.float32]] = []
                for rel in ply_rels:
                    p = self.root / rel
                    if not p.exists():
                        continue
                    pts = load_ply_xyz(p)
                    M = scan_alignment.get(p.name)
                    if M is not None and pts.size > 0:
                        R = M[:3, :3]
                        t = M[:3, 3]
                        pts = (pts.astype(np.float64) @ R.T + t).astype(np.float32)
                    chunks.append(pts)
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
                    self._gt_cache[cache_key] = pcd

        if self.with_per_view_gt and ply_rels:
            scene_pv = self._load_or_render_scene_pv_depth(
                scene=rec["scene"],
                ply_rels=ply_rels,
            )
            if scene_pv is not None:
                # Pre-render canvas: native unless mono-depth asks for alignment.
                if self.resize_images_to_pv_render:
                    render_sizes = [
                        scene_pv["render_sizes"][
                            scene_pv["image_id_to_idx"][int(ir["image_id"])]
                        ]
                        for ir in rec["image_records"]
                        if int(ir["image_id"]) in scene_pv["image_id_to_idx"]
                    ]
                    if render_sizes:
                        canvas_h = max(h for h, _ in render_sizes)
                        canvas_w = max(w for _, w in render_sizes)
                    else:
                        canvas_h = max(img.shape[0] for img in images)
                        canvas_w = max(img.shape[1] for img in images)
                else:
                    canvas_h = max(img.shape[0] for img in images)
                    canvas_w = max(img.shape[1] for img in images)
                depth_gt, depth_valid, gt_sizes = self._stack_per_view_depth_for_sample(
                    scene_pv=scene_pv,
                    image_records=rec["image_records"],
                    canvas_h=canvas_h,
                    canvas_w=canvas_w,
                )

        if self.resize_images_to_pv_render and gt_sizes is not None:
            resized: list[NDArray[np.uint8]] = []
            for img, (H_r, W_r) in zip(images, gt_sizes, strict=True):
                if H_r > 0 and W_r > 0:
                    resized.append(_resize_rgb_uint8(img, height=H_r, width=W_r))
                else:
                    resized.append(img)
            images = resized

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

        metadata: dict[str, Any] = {
            "scene": rec["scene"],
            "native_sizes": native_sizes,
            "split": self.split,
        }
        if gt_sizes is not None:
            metadata["gt_sizes"] = gt_sizes
        if self.resize_images_to_pv_render and gt_sizes is not None:
            metadata["eval_sizes"] = gt_sizes
        return Sample(
            sample_id=rec["sample_id"],
            images=padded,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            point_cloud_gt=pcd,
            metadata=metadata,
        )

    def _load_or_render_scene_pv_depth(
        self,
        *,
        scene: str,
        ply_rels: list[str],
    ) -> dict[str, Any] | None:
        """Render per-view GT depth for every image in a scene, once.

        Concatenates the (MLP-aligned) laser PLY into a single scene
        cloud, then for each image in the scene's `images.txt`
        z-buffers the cloud through that image's camera at the image's
        native resolution (capped at ``pv_render_max_dim``). Saves to
        disk as one npz per scene. Returns a dict shaped::

            {
              "depths":  (V_scene, max_H_render, max_W_render) float32,
              "valids":  (V_scene, max_H_render, max_W_render) bool,
              "render_sizes": [(H_render, W_render), ...] per image,
              "image_id_to_idx": {colmap_image_id: index},
            }
        """
        if scene in self._pv_depth_cache:
            return self._pv_depth_cache[scene]

        cache_path = (
            self.root / ".plumbline_manifest"
            / f"eth3d_pv_depth_{scene}_max{self.pv_render_max_dim}_r{self.pv_splat_radius}.npz"
        )
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as f:
                cached = {
                    "depths": f["depths"].astype(np.float32),
                    "valids": f["valids"].astype(np.bool_),
                    "render_sizes": [tuple(map(int, x)) for x in f["render_sizes"]],
                    "image_id_to_idx": {int(k): int(v) for k, v in f["image_ids"]},
                }
                self._pv_depth_cache[scene] = cached
                return cached

        # Need to render. Locate the scene dir + parse calibration.
        scene_dir = self.root / scene
        calib = scene_dir / "dslr_calibration_undistorted"
        if not (calib / "images.txt").exists():
            return None
        cameras = parse_colmap_cameras(calib / "cameras.txt")
        images_info = sorted(
            parse_colmap_images(calib / "images.txt"),
            key=lambda x: x["image_id"],
        )

        # Load + MLP-transform the scene PLY (full density; do NOT
        # subsample here — sparse GT renders give bad coverage).
        scan_alignment: dict[str, NDArray[np.float64]] = {}
        if ply_rels:
            mlp_dir = (self.root / ply_rels[0]).parent
            mlp_path = mlp_dir / "scan_alignment.mlp"
            if mlp_path.exists():
                scan_alignment = parse_scan_alignment_mlp(mlp_path)
        chunks: list[NDArray[np.float32]] = []
        for rel in ply_rels:
            p = self.root / rel
            if not p.exists():
                continue
            pts = load_ply_xyz(p)
            M = scan_alignment.get(p.name)
            if M is not None and pts.size > 0:
                R = M[:3, :3]
                t = M[:3, 3]
                pts = (pts.astype(np.float64) @ R.T + t).astype(np.float32)
            chunks.append(pts)
        if not chunks:
            return None
        xyz = np.concatenate(chunks, axis=0)

        # Per-view render. Each image's native size is in
        # ``images.txt``'s associated cameras.txt entry; we use its
        # native (H, W) from cameras and scale to render_max_dim.
        from plumbline.datasets.dtu import render_pv_depth_zbuffer

        # For each image: K_native, world_from_cam, native (H, W).
        # cameras.txt provides per-camera-id (H, W) and K — we already
        # parse it. Fetch the (H, W) by re-reading raw lines.
        cam_native_hw = _parse_colmap_cameras_hw(calib / "cameras.txt")
        depths_list: list[NDArray[np.float32]] = []
        valids_list: list[NDArray[np.bool_]] = []
        render_sizes: list[tuple[int, int]] = []
        image_id_to_idx: dict[int, int] = {}
        max_render_h = 0
        max_render_w = 0
        for idx, ir in enumerate(images_info):
            cam_id = ir["camera_id"]
            H_native, W_native = cam_native_hw[cam_id]
            scale = 1.0
            if self.pv_render_max_dim is not None:
                scale = min(1.0, self.pv_render_max_dim / max(H_native, W_native))
            H_render = max(1, int(round(H_native * scale)))
            W_render = max(1, int(round(W_native * scale)))
            K_native = cameras[cam_id].astype(np.float64)
            K_render = K_native.copy()
            K_render[0, :] *= W_render / W_native
            K_render[1, :] *= H_render / H_native
            q = np.array([ir["qw"], ir["qx"], ir["qy"], ir["qz"]], dtype=np.float64)
            t = np.array([ir["tx"], ir["ty"], ir["tz"]], dtype=np.float64)
            cam_from_world = np.eye(4, dtype=np.float64)
            cam_from_world[:3, :3] = quat_to_rot(q)
            cam_from_world[:3, 3] = t
            d, v = render_pv_depth_zbuffer(
                xyz, K_render, cam_from_world, H_render, W_render,
                splat_radius=self.pv_splat_radius,
            )
            depths_list.append(d)
            valids_list.append(v)
            render_sizes.append((H_render, W_render))
            image_id_to_idx[ir["image_id"]] = idx
            max_render_h = max(max_render_h, H_render)
            max_render_w = max(max_render_w, W_render)

        # Pad to common (max_render_h, max_render_w) canvas; padding is
        # zero depth / False valid, matching the per-view-masked path's
        # masking expectation.
        V = len(depths_list)
        depths = np.zeros((V, max_render_h, max_render_w), dtype=np.float32)
        valids = np.zeros((V, max_render_h, max_render_w), dtype=np.bool_)
        for i, (d, v, (H_r, W_r)) in enumerate(zip(depths_list, valids_list, render_sizes, strict=True)):
            depths[i, :H_r, :W_r] = d
            valids[i, :H_r, :W_r] = v

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            depths=depths,
            valids=valids,
            render_sizes=np.asarray(render_sizes, dtype=np.int32),
            image_ids=np.asarray(list(image_id_to_idx.items()), dtype=np.int64),
        )
        cached = {
            "depths": depths,
            "valids": valids,
            "render_sizes": render_sizes,
            "image_id_to_idx": image_id_to_idx,
        }
        self._pv_depth_cache[scene] = cached
        return cached

    def _stack_per_view_depth_for_sample(
        self,
        *,
        scene_pv: dict[str, Any],
        image_records: list[dict[str, Any]],
        canvas_h: int,
        canvas_w: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], list[tuple[int, int]]]:
        """Slice the scene-level rendered depth into per-sample tensors
        padded to ``sample.images``' canvas. Returns ``(depth, valid,
        gt_sizes)`` where ``gt_sizes`` records each view's actual
        depth-render extent within the canvas (which can differ from
        the view's image-native size when the loader rendered at a
        capped resolution); the runner uses it to NN-sample correctly
        while still rescaling K from image-native."""
        depths_scene: NDArray[np.float32] = scene_pv["depths"]
        valids_scene: NDArray[np.bool_] = scene_pv["valids"]
        render_sizes: list[tuple[int, int]] = scene_pv["render_sizes"]
        image_id_to_idx: dict[int, int] = scene_pv["image_id_to_idx"]

        V = len(image_records)
        out_d = np.zeros((V, canvas_h, canvas_w), dtype=np.float32)
        out_v = np.zeros((V, canvas_h, canvas_w), dtype=np.bool_)
        gt_sizes: list[tuple[int, int]] = []
        for i, ir in enumerate(image_records):
            idx = image_id_to_idx.get(int(ir["image_id"]))
            if idx is None:
                gt_sizes.append((0, 0))
                continue
            H_r, W_r = render_sizes[idx]
            # If render is larger than image canvas (shouldn't happen with
            # pv_render_max_dim ≤ image native), clip to canvas to avoid
            # an index error; loses peripheral coverage but doesn't crash.
            H_r_eff = min(H_r, canvas_h)
            W_r_eff = min(W_r, canvas_w)
            out_d[i, :H_r_eff, :W_r_eff] = depths_scene[idx, :H_r_eff, :W_r_eff]
            out_v[i, :H_r_eff, :W_r_eff] = valids_scene[idx, :H_r_eff, :W_r_eff]
            gt_sizes.append((H_r_eff, W_r_eff))
        return out_d, out_v, gt_sizes


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


def _parse_colmap_cameras_hw(path: Path) -> dict[int, tuple[int, int]]:
    """Read just (H, W) per camera_id from a COLMAP ``cameras.txt``.

    Format: ``CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]``. Used by the
    per-view-GT renderer to size each view's output before scaling
    to ``pv_render_max_dim``. Returns ``{camera_id: (H, W)}``.
    """
    out: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            cam_id = int(parts[0])
            W = int(parts[2])
            H = int(parts[3])
            out[cam_id] = (H, W)
    return out


def parse_scan_alignment_mlp(path: Path) -> dict[str, NDArray[np.float64]]:
    """Parse an ETH3D ``scan_alignment.mlp`` (MeshLab project) file.

    Returns ``{scan_filename: 4x4 transform}`` mapping each PLY's
    filename to the 4x4 ``MLMatrix44`` that brings its points from
    scanner-local coordinates into the scene's COLMAP/DSLR world
    frame. ETH3D ships these per scene; without them, ``scan1.ply``
    and ``scan2.ply`` live in different scanner frames and a naive
    concatenation produces a rotationally-misaligned GT cloud.

    The file is small XML; we use the stdlib parser instead of pulling
    a dependency. Schema (per ETH3D)::

        <MeshLabProject>
          <MeshGroup>
            <MLMesh label="scan2.ply" filename="scan2.ply">
              <MLMatrix44>
                m00 m01 m02 m03
                m10 m11 m12 m13
                m20 m21 m22 m23
                0   0   0   1
              </MLMatrix44>
            </MLMesh>
            ...
    """
    import xml.etree.ElementTree as ET

    out: dict[str, NDArray[np.float64]] = {}
    tree = ET.parse(path)
    for mesh in tree.iterfind(".//MLMesh"):
        fname = mesh.get("filename") or mesh.get("label") or ""
        m_node = mesh.find("MLMatrix44")
        if not fname or m_node is None or not m_node.text:
            continue
        nums = [float(x) for x in m_node.text.split()]
        if len(nums) != 16:
            continue
        out[fname] = np.asarray(nums, dtype=np.float64).reshape(4, 4)
    return out


def _resolve_scan_clean_plys(scene_dir: Path) -> list[Path]:
    """Find the ground-truth laser scan ply files for a scene.

    ETH3D ships two GT scan variants per scene:

    - ``scan_clean/`` — the raw laser scan, broader spatial extent than
      what the DSLR cameras could possibly see.
    - ``dslr_scan_eval/`` — the same scan clipped to the DSLR-visible
      frustum. This is what ETH3D's official evaluation protocol uses,
      and it matches the coverage of monocular-depth predictions.

    For MVS / chamfer evaluation, ``dslr_scan_eval`` is correct —
    otherwise GT points outside the camera frustum inflate completeness
    (GT→pred nearest-distance), which is how D4 landed 2× worse vs the
    prior run on 2026-04-24 when the loader was swapped to ``scan_clean``
    without this clipping.

    Prefer ``dslr_scan_eval`` when present; fall back to ``scan_clean``
    (subdir or single-file) for older layouts. Older manually-prepared
    mirrors sometimes placed a single ``scan_clean.ply`` at the scene
    root. Return a sorted list of paths, or an empty list.
    """
    dslr = scene_dir / "dslr_scan_eval"
    if dslr.is_dir():
        plys = sorted(dslr.glob("scan*.ply"))
        if plys:
            return plys
    direct = scene_dir / "scan_clean.ply"
    if direct.exists():
        return [direct]
    subdir = scene_dir / "scan_clean"
    if subdir.is_dir():
        return sorted(subdir.glob("scan*.ply"))
    return []


# _load_ply_xyz lived here previously. It was moved to
# plumbline.datasets._common.load_ply_xyz so the DTU loader could share
# it; this module now re-exports the canonical version for backward
# compatibility with any external caller that imported the private name.
_load_ply_xyz = load_ply_xyz
