"""Co3Dv2 pose-benchmark loader.

Common Objects in 3D v2 (Meta; Reizenstein et al. 2021) is a public
object-centric video dataset with GT camera poses per frame. VGGT
Table 1, DUSt3R, and MASt3R all report multi-view pose benchmarks on
Co3Dv2 — plumbline's primary Tier-2 multi-view pose-benchmark target
(public, no ToS gate).

Expected layout (point ``--data-root`` or ``$CO3DV2_ROOT`` here)::

    <root>/
      <category>/
        <sequence>/
          images/frame000001.jpg ...
          depths/frame000001.png  (optional, SfM-sparse)
          masks/frame000001.png   (optional, fg/bg)
        frame_annotations.jgz     (per-category metadata)
        sequence_annotations.jgz
        set_lists/
          set_lists_manyview_dev.json
          ...
        eval_batches/
          eval_batches_manyview_dev.json
          ...

Full Co3Dv2 is ~5.5 TB; the single-sequence subset (used for
many-view eval) is ~8.9 GB. For pose benchmarks, papers typically
evaluate on a handful of sequences per category — the loader accepts
an explicit ``sequences`` whitelist so you can pin a small eval
split without carrying the full tarball.

Conventions
-----------
- Upstream frame_annotations use PyTorch3D conventions:
    * Extrinsics: ``X_cam = X_world @ R + T`` (right-multiply).
    * Intrinsics: NDC-style, ``ndc_norm_image_bounds`` by default —
      the image is in ``[-1, 1] x [-1, 1]``, NOT pixel space.
- plumbline uses OpenCV conventions:
    * Extrinsics: ``world_from_camera`` column-vector form.
    * Intrinsics: pixel-space K with principal point at image centre.
- The loader converts both at load time.

References
----------
https://github.com/facebookresearch/co3d
"""

from __future__ import annotations

import gzip
import json
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
    read_rgb_uint8,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["Co3Dv2Dataset", "co3d_pytorch3d_to_opencv"]


@register_dataset("co3dv2")
class Co3Dv2Dataset(Dataset):
    """Co3Dv2 object-centric pose-benchmark loader.

    Each :class:`Sample` is one sequence's N-frame sliding window (or
    all frames if the sequence is short enough and ``views_per_sample``
    ≥ sequence length), with images + per-frame intrinsics (pixel K)
    + GT world_from_camera extrinsics. Depth is **not** returned by
    default — Co3Dv2's depth is SfM-sparse and not a canonical
    depth-eval target; the primary benchmark use is pose.

    Parameters
    ----------
    root
        Dataset root containing ``<category>/<sequence>/...``. Falls
        back to ``$CO3DV2_ROOT``.
    categories
        Optional list of category names to include (e.g.
        ``["hydrant", "teddybear"]``). ``None`` = all categories on disk.
    sequences
        Optional list of sequence names to restrict to (across all
        kept categories). Useful for pinning a tiny eval subset.
    views_per_sample
        Number of consecutive frames per :class:`Sample`. Paper pose
        benchmarks typically use 2-10 views per sample.
    max_sequences_per_category
        Cap the number of sequences drawn from each category. Helps
        keep a pose reproduction bounded when operating on a large
        Co3Dv2 dump.
    frame_stride
        Sample every Nth frame within a sequence (default 1 = every
        frame). Larger strides widen the pose baselines within the
        per-sample view set.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        categories: list[str] | None = None,
        sequences: list[str] | None = None,
        views_per_sample: int = 4,
        max_sequences_per_category: int | None = None,
        frame_stride: int = 1,
    ) -> None:
        root_path = Path(root) if root else env_path("CO3DV2_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "Co3Dv2 not found. Set --data-root or $CO3DV2_ROOT to a directory "
                "containing <category>/<sequence>/images/*.jpg and "
                "<category>/frame_annotations.jgz. Download script: "
                "https://github.com/facebookresearch/co3d (full ~5.5 TB; "
                "single-sequence subset ~8.9 GB via --single_sequence_subset)."
            )
        if views_per_sample < 1:
            raise ValueError(f"views_per_sample must be >= 1; got {views_per_sample}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1; got {frame_stride}")

        self.root = root_path
        self.views_per_sample = int(views_per_sample)
        self.frame_stride = int(frame_stride)
        self.max_sequences_per_category = max_sequences_per_category
        self._sequences_whitelist = set(sequences) if sequences else None

        # Enumerate categories on disk.
        all_categories = sorted(
            p.name
            for p in root_path.iterdir()
            if p.is_dir() and (p / "frame_annotations.jgz").exists()
        )
        if categories is not None:
            kept = [c for c in categories if c in set(all_categories)]
            missing = set(categories) - set(kept)
            if missing:
                raise DatasetNotAvailable(
                    f"Categories not found on disk: {sorted(missing)}. "
                    f"Available: {all_categories[:10]}{'...' if len(all_categories) > 10 else ''}"
                )
            self.categories = kept
        else:
            self.categories = all_categories
        if not self.categories:
            raise DatasetNotAvailable(
                f"No Co3Dv2 categories found under {root_path}. Each category "
                "directory must contain a frame_annotations.jgz file."
            )

        self._records = list(self._build_records())

    # -- scanning --------------------------------------------------------

    def _build_records(self) -> Iterator[dict[str, Any]]:
        for category in self.categories:
            anno_path = self.root / category / "frame_annotations.jgz"
            by_sequence: dict[str, list[dict[str, Any]]] = {}
            with gzip.open(anno_path, "rt", encoding="utf-8") as f:
                annotations = json.load(f)
            for frame in annotations:
                seq = frame["sequence_name"]
                if self._sequences_whitelist and seq not in self._sequences_whitelist:
                    continue
                by_sequence.setdefault(seq, []).append(frame)
            sequence_names = sorted(by_sequence.keys())
            if self.max_sequences_per_category is not None:
                sequence_names = sequence_names[: self.max_sequences_per_category]
            for seq in sequence_names:
                frames = sorted(by_sequence[seq], key=lambda x: int(x["frame_number"]))
                if len(frames) < self.views_per_sample:
                    continue
                strided = frames[:: self.frame_stride]
                for i in range(0, len(strided) - self.views_per_sample + 1):
                    group = strided[i : i + self.views_per_sample]
                    yield {
                        "category": category,
                        "sequence": seq,
                        "frame_numbers": [int(f["frame_number"]) for f in group],
                        "frames": group,
                    }

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        frames = rec["frames"]
        images: list[NDArray[np.uint8]] = []
        Ks: list[NDArray[np.float32]] = []
        world_from_cam_list: list[NDArray[np.float64]] = []
        for frame in frames:
            img_path = self.root / frame["image"]["path"]
            img = read_rgb_uint8(img_path)
            # Per-frame image size is `(H, W)` per Co3D convention.
            H, W = img.shape[:2]
            images.append(img)
            vp = frame["viewpoint"]
            K = co3d_ndc_intrinsics_to_pixel(
                focal_length=tuple(vp["focal_length"]),
                principal_point=tuple(vp["principal_point"]),
                size_hw=(H, W),
                intrinsics_format=vp.get("intrinsics_format", "ndc_norm_image_bounds"),
            )
            Ks.append(K.astype(np.float32))
            # PyTorch3D: X_cam = X_world @ R + T. Convert to
            # world_from_cam OpenCV via co3d_pytorch3d_to_opencv.
            R = np.asarray(vp["R"], dtype=np.float64)
            T = np.asarray(vp["T"], dtype=np.float64)
            world_from_cam_list.append(co3d_pytorch3d_to_opencv(R, T))

        # Co3D sequences may have slightly varying image sizes between
        # frames (rare but possible). Stack only when all the same; else
        # pad to max.
        sizes = {img.shape for img in images}
        if len(sizes) == 1:
            images_arr = np.stack(images, axis=0)
        else:
            max_h = max(img.shape[0] for img in images)
            max_w = max(img.shape[1] for img in images)
            images_arr = np.zeros((len(images), max_h, max_w, 3), dtype=np.uint8)
            for i, img in enumerate(images):
                h, w, _ = img.shape
                images_arr[i, :h, :w] = img
        assert_valid_image(images_arr, name=f"co3dv2/{rec['sequence']}/image")

        K_stack = np.stack(Ks, axis=0)
        world_from_cam = np.stack(world_from_cam_list, axis=0)
        # Rebase: plumbline convention is first-camera-as-world.
        extrinsics = rebase_to_first_camera(world_from_cam).astype(np.float32)

        assert_valid_intrinsics(K_stack, name=f"co3dv2/{rec['sequence']}/intrinsics")
        assert_valid_extrinsics(extrinsics, name=f"co3dv2/{rec['sequence']}/extrinsics")

        return Sample(
            sample_id=f"{rec['category']}/{rec['sequence']}/f{rec['frame_numbers'][0]:06d}_v{self.views_per_sample}",
            images=images_arr,
            intrinsics=K_stack,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={
                "category": rec["category"],
                "sequence": rec["sequence"],
                "frame_numbers": rec["frame_numbers"],
                "split": self.split,
            },
        )


# ---------------------------------------------------------------------------
# Coordinate + intrinsic conversions
# ---------------------------------------------------------------------------


def co3d_pytorch3d_to_opencv(
    R_p3d: NDArray[np.float64], T_p3d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert Co3D's PyTorch3D camera params to OpenCV ``world_from_camera``.

    PyTorch3D convention (right-multiply, row vector):
        ``X_cam_p3d = X_world @ R + T``

    with the additional PyTorch3D-vs-OpenCV axis flip: PyTorch3D has
    +X-left, +Y-up (image origin bottom-left), while OpenCV has +X-right,
    +Y-down (image origin top-left). Both share +Z-forward.

    To convert:
        1. Rewrite right-multiply as column-form:
               X_cam_p3d = R.T @ X_world + T
           so ``cam_from_world_p3d = [R.T | T]``.
        2. Flip the X, Y axes to get OpenCV cam:
               X_cam_ocv = diag(-1, -1, 1) @ X_cam_p3d
        3. Combine and invert to ``world_from_camera``.
    """
    flip = np.diag([-1.0, -1.0, 1.0])
    R_cam_from_world = flip @ R_p3d.T
    t_cam_from_world = flip @ T_p3d
    cam_from_world = np.eye(4, dtype=np.float64)
    cam_from_world[:3, :3] = R_cam_from_world
    cam_from_world[:3, 3] = t_cam_from_world
    return invert_pose(cam_from_world)


def co3d_ndc_intrinsics_to_pixel(
    *,
    focal_length: tuple[float, float],
    principal_point: tuple[float, float],
    size_hw: tuple[int, int],
    intrinsics_format: str = "ndc_norm_image_bounds",
) -> NDArray[np.float64]:
    """Convert Co3D NDC-style intrinsics to pixel-space K (3x3).

    Co3D stores focal + principal point in one of two NDC conventions:

      - ``ndc_norm_image_bounds`` (default): the image spans
        ``[-1, 1] x [-1, 1]`` regardless of aspect ratio. x / y have
        independent pixel-per-ndc scales.
      - ``ndc_isotropic``: PyTorch3D 0.5+. The shorter image side has
        range ``[-1, 1]``; the longer side has range ``[-s, s]`` where
        ``s = max(H, W) / min(H, W)``. Same pixel-per-ndc scale along
        both axes.

    plumbline wants pixel-space K where the principal point is in
    pixel coords measured from the top-left origin. Convert by:
      1. Map NDC centre (0, 0) → pixel ``(W/2, H/2)``.
      2. NDC vector (1, 0) in x maps to pixel ``(W/2, 0)`` (OR
         ``(max_side/2, 0)`` in isotropic mode).
      3. Also flip y because Co3D / PyTorch3D's NDC has +y UP while
         OpenCV pixel-space has +y DOWN.
    """
    H, W = size_hw
    fx_ndc, fy_ndc = focal_length
    cx_ndc, cy_ndc = principal_point
    if intrinsics_format == "ndc_norm_image_bounds":
        # Each side half-span = H/2 pixels (y) or W/2 pixels (x).
        fx_px = fx_ndc * (W / 2.0)
        fy_px = fy_ndc * (H / 2.0)
        cx_px = W / 2.0 - cx_ndc * (W / 2.0)
        # Flip y: +y up in NDC → +y down in OpenCV pixels
        cy_px = H / 2.0 - cy_ndc * (H / 2.0)
    elif intrinsics_format == "ndc_isotropic":
        half = max(H, W) / 2.0
        fx_px = fx_ndc * half
        fy_px = fy_ndc * half
        cx_px = W / 2.0 - cx_ndc * half
        cy_px = H / 2.0 - cy_ndc * half
    else:
        raise ValueError(
            f"unknown Co3D intrinsics_format {intrinsics_format!r}; expected "
            "'ndc_norm_image_bounds' or 'ndc_isotropic'"
        )
    K = np.array(
        [[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K
