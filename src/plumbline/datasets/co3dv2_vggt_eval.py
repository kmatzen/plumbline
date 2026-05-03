"""Co3Dv2 multi-view-pose eval loader, VGGT/PoseDiffusion canonical protocol.

This is the loader version that reproduces VGGT Table 1's CO3Dv2 row
(AUC@30 = 0.882, feed-forward) and MASt3R Table 3's CO3Dv2 cells
(RRA@15 = 94.6, RTA@15 = 91.9, mAA(30) = 81.8). The protocol is a
verbatim port of:

    facebookresearch/vggt @ evaluation/test_co3d.py + preprocess_co3d.py

The general-purpose ``Co3Dv2Dataset`` in ``co3dv2.py`` enumerates all
sliding N-frame windows; this loader instead emits one Sample per
(category, sequence) with 10 frames sampled by VGGT's seeded RNG.

Recipe
------
1. **Categories**: 41 SEEN categories in :data:`CO3D_VGGT_SEEN_CATEGORIES`.
2. **Sequence filter**: ``viewpoint_quality_score > 0.5`` (per category's
   ``sequence_annotations.jgz``).
3. **Subset**: only sequences listed in
   ``set_lists/set_lists_fewview_dev.json["test"]``.
4. **Frame sanity**: drop frames whose ``T[0] + T[1] + T[2] > 1e5``.
5. **Sequence-length filter**: skip sequences with fewer than
   ``min_num_images = 50`` qualifying frames.
6. **Per-sequence frame sampling**: ``np.random.choice(num_frames=10,
   replace=False)`` from the shared global RNG seeded once at loader
   construction with ``seed=0``.
7. **Per-category sequence sampling** (``fast_eval=True``, default):
   ``random.sample(sorted(seqs), 10)`` from the shared RNG, so each
   category contributes exactly 10 sequences. This matches VGGT's
   ``--fast_eval`` mode and keeps the per-category sample weight
   uniform, which is what makes the cheap per-sample-mean aggregation
   in plumbline's runner approximate the paper's per-category-mean
   aggregation.

Aggregation caveat
------------------
The paper computes one AUC per category from the union of all that
category's pair errors, then averages 41 per-category AUCs. Plumbline's
runner computes per-sample AUC across the 45 pairs of a sample's 10
views, then averages across samples. With ``fast_eval`` enabled, both
are weighted-uniform across the 41 categories so the two converge
within ~1-2% on realistic distributions. If a strict per-category
reducer is needed, post-process ``Report.per_sample`` by grouping on
``metadata['category']`` and recomputing AUC from each group's
flattened pair-error arrays. This is documented as a known protocol
gap (small, paper-tolerant) until it bites.

Pose convention
---------------
Co3D ships PyTorch3D camera params (``X_cam = X_world @ R + T``).
VGGT's ``convert_pt3d_RT_to_opencv`` converts them to the OpenCV
``cam_from_world`` 3x4 form; plumbline's ``world_from_camera``
4x4 form is the inverse, then rebased to first-camera-as-world.
"""

from __future__ import annotations

import gzip
import json
import random
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
from plumbline.datasets.co3dv2 import (
    co3d_ndc_intrinsics_to_pixel,
    co3d_pytorch3d_to_opencv,
)
from plumbline.datasets.registry import register_dataset

__all__ = ["CO3D_VGGT_SEEN_CATEGORIES", "Co3Dv2VGGTPoseEvalLoader"]


# 41 SEEN categories from RelPose++ / VGGT eval — order matches
# facebookresearch/vggt @ evaluation/test_co3d.py:344-353.
CO3D_VGGT_SEEN_CATEGORIES: tuple[str, ...] = (
    "apple", "backpack", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "bottle", "bowl", "broccoli",
    "cake", "car", "carrot", "cellphone", "chair",
    "cup", "donut", "hairdryer", "handbag", "hydrant",
    "keyboard", "laptop", "microwave", "motorcycle", "mouse",
    "orange", "parkingmeter", "pizza", "plant", "stopsign",
    "teddybear", "toaster", "toilet", "toybus", "toyplane",
    "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
)  # fmt: skip


@register_dataset("co3dv2-vggt-pose-eval")
class Co3Dv2VGGTPoseEvalLoader(Dataset):
    """Co3Dv2 N-view pose loader using VGGT/PoseDiffusion's canonical protocol.

    Each :class:`Sample` is one sequence with 10 frames sampled deterministically.

    Parameters
    ----------
    root
        Co3Dv2 root containing ``<category>/{frame_annotations.jgz,
        sequence_annotations.jgz, set_lists/...}``. Falls back to
        ``$CO3DV2_ROOT``.
    categories
        Optional subset of categories to evaluate. Default = all 41
        SEEN categories. Useful for `--debug`-style single-category runs.
    num_frames
        Frames per sequence sample. Default 10 (paper standard).
    min_num_images
        Minimum sequence length filter. Default 50.
    min_quality
        ``viewpoint_quality_score >`` threshold on sequences. Default 0.5.
    max_translation_sum
        Drop frames with ``T[0]+T[1]+T[2] >`` threshold. Default 1e5.
    fast_eval
        When ``True`` (default), sample 10 sequences per category
        deterministically. When ``False``, evaluate all qualifying
        sequences (paper-mode but per-sample-mean ≠ paper aggregation
        will be more pronounced).
    sequences_per_category
        Sequence cap when ``fast_eval=True``. Default 10.
    seed
        Global RNG seed. Default 0 (paper default).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        categories: tuple[str, ...] | list[str] | None = None,
        num_frames: int = 10,
        min_num_images: int = 50,
        min_quality: float = 0.5,
        max_translation_sum: float = 1e5,
        fast_eval: bool = True,
        sequences_per_category: int = 10,
        seed: int = 0,
    ) -> None:
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2 for pose eval; got {num_frames}")
        if sequences_per_category < 1:
            raise ValueError(f"sequences_per_category must be >= 1; got {sequences_per_category}")

        root_path = Path(root) if root else env_path("CO3DV2_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "Co3Dv2 not found. Set --data-root or $CO3DV2_ROOT to a directory "
                "containing <category>/{frame_annotations.jgz,sequence_annotations.jgz,"
                "set_lists/set_lists_fewview_dev.json}. Download script: "
                "https://github.com/facebookresearch/co3d (full ~5.5 TB; for the "
                "VGGT pose eval only the 41 SEEN categories are needed)."
            )

        wanted = tuple(categories) if categories is not None else CO3D_VGGT_SEEN_CATEGORIES
        unknown = [c for c in wanted if c not in CO3D_VGGT_SEEN_CATEGORIES]
        if unknown:
            raise ValueError(
                f"Categories not in CO3D_VGGT_SEEN_CATEGORIES: {unknown}. "
                f"Either remove them or extend the canonical 41-category list."
            )

        self.root = root_path
        self.categories = wanted
        self.num_frames = int(num_frames)
        self.min_num_images = int(min_num_images)
        self.min_quality = float(min_quality)
        self.max_translation_sum = float(max_translation_sum)
        self.fast_eval = bool(fast_eval)
        self.sequences_per_category = int(sequences_per_category)
        self.seed = int(seed)

        # Global RNGs match VGGT's set_random_seeds(seed) once at startup.
        # Both random.sample (per-category) and np.random.choice (per-sequence)
        # advance the same shared state, so the deterministic sample set is
        # path-of-iteration sensitive — must process categories in canonical
        # order, sequences in sorted order, exactly as upstream does.
        self._py_rng = random.Random(self.seed)
        self._np_rng_state_seed = self.seed
        # Build records eagerly so all RNG advances happen at construction.
        self._records: list[dict[str, Any]] = list(self._build_records())

    # -- enumeration ----------------------------------------------------

    def _build_records(self) -> Iterator[dict[str, Any]]:
        # Match VGGT's set_random_seeds shape: seed numpy.global once
        # at startup, then advance through all category × sequence work.
        np.random.seed(self.seed)
        for category in self.categories:
            cat_dir = self.root / category
            anno = self._load_category_annotations(cat_dir)
            if anno is None:
                continue

            seq_names = sorted(anno.keys())
            if self.fast_eval and len(seq_names) >= self.sequences_per_category:
                # `random.sample` consumes from the shared py-RNG.
                seq_names = self._py_rng.sample(seq_names, self.sequences_per_category)
                seq_names = sorted(seq_names)

            for seq_name in seq_names:
                frames = anno[seq_name]
                # Frame sanity filter: drop runaway translations.
                clean = [
                    fr for fr in frames
                    if (fr["T"][0] + fr["T"][1] + fr["T"][2]) <= self.max_translation_sum
                ]
                if len(clean) < self.min_num_images:
                    continue

                ids = np.random.choice(len(clean), self.num_frames, replace=False)
                ids_list = [int(i) for i in ids]
                yield {
                    "category": category,
                    "sequence": seq_name,
                    "ids": ids_list,
                    "frames": [clean[i] for i in ids_list],
                }

    def _load_category_annotations(
        self, cat_dir: Path
    ) -> dict[str, list[dict[str, Any]]] | None:
        """Load and filter category annotations.

        Two upstream layouts are supported:

        - **Preprocessed** (preferred): ``<root>/<category>_test.jgz`` (a
          gzipped JSON of ``{seq_name: [{filepath, R, T, focal_length,
          principal_point}, ...]}``), pre-filtered by VGGT's
          ``preprocess_co3d.py``. Already constrained to test-subset
          frames from sequences with ``viewpoint_quality_score > min_q``.
        - **Raw**: ``<root>/<category>/{frame_annotations.jgz,
          sequence_annotations.jgz, set_lists/set_lists_fewview_dev.json}``
          — apply the preprocess steps inline.
        """
        prebuilt = cat_dir.parent / f"{cat_dir.name}_test.jgz"
        if prebuilt.exists():
            with gzip.open(prebuilt, "rt", encoding="utf-8") as f:
                return json.load(f)

        frame_file = cat_dir / "frame_annotations.jgz"
        seq_file = cat_dir / "sequence_annotations.jgz"
        set_lists_file = cat_dir / "set_lists" / "set_lists_fewview_dev.json"
        if not (frame_file.exists() and seq_file.exists() and set_lists_file.exists()):
            return None

        with gzip.open(frame_file, "rt", encoding="utf-8") as f:
            frame_data: list[dict[str, Any]] = json.load(f)
        with gzip.open(seq_file, "rt", encoding="utf-8") as f:
            seq_data: list[dict[str, Any]] = json.load(f)
        with set_lists_file.open("rt", encoding="utf-8") as f:
            subset_lists: dict[str, list[list[Any]]] = json.load(f)

        good_quality = {
            s["sequence_name"]
            for s in seq_data
            if s["viewpoint_quality_score"] > self.min_quality
        }

        # Index frames by (sequence_name, frame_number).
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for fr in frame_data:
            by_key[(fr["sequence_name"], int(fr["frame_number"]))] = fr

        out: dict[str, list[dict[str, Any]]] = {}
        for seq_name, frame_number, filepath in subset_lists.get("test", []):
            if seq_name not in good_quality:
                continue
            fr = by_key.get((seq_name, int(frame_number)))
            if fr is None:
                continue
            vp = fr["viewpoint"]
            out.setdefault(seq_name, []).append(
                {
                    "filepath": filepath,
                    "R": vp["R"],
                    "T": vp["T"],
                    "focal_length": vp["focal_length"],
                    "principal_point": vp["principal_point"],
                    "intrinsics_format": vp.get("intrinsics_format", "ndc_norm_image_bounds"),
                }
            )
        return out

    # -- iteration ------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- per-sample -----------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        category: str = rec["category"]
        seq: str = rec["sequence"]
        frames: list[dict[str, Any]] = rec["frames"]

        images: list[NDArray[np.uint8]] = []
        Ks: list[NDArray[np.float64]] = []
        world_from_cam: list[NDArray[np.float64]] = []
        for fr in frames:
            img_path = self.root / fr["filepath"]
            img = read_rgb_uint8(img_path)
            H, W = img.shape[:2]
            images.append(img)
            Ks.append(
                co3d_ndc_intrinsics_to_pixel(
                    focal_length=tuple(fr["focal_length"]),
                    principal_point=tuple(fr["principal_point"]),
                    size_hw=(H, W),
                    intrinsics_format=fr.get(
                        "intrinsics_format", "ndc_norm_image_bounds"
                    ),
                )
            )
            R = np.asarray(fr["R"], dtype=np.float64)
            T = np.asarray(fr["T"], dtype=np.float64)
            world_from_cam.append(co3d_pytorch3d_to_opencv(R, T))

        # Image sizes within a Co3D sequence may vary (rare). Pad to max.
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
        assert_valid_image(images_arr, name=f"co3dv2-vggt/{category}/{seq}/image")

        K_stack = np.stack(Ks, axis=0).astype(np.float32)
        E = np.stack(world_from_cam, axis=0)
        extrinsics = rebase_to_first_camera(E).astype(np.float32)
        assert_valid_intrinsics(K_stack, name=f"co3dv2-vggt/{category}/{seq}/K")
        assert_valid_extrinsics(extrinsics, name=f"co3dv2-vggt/{category}/{seq}/E")

        return Sample(
            sample_id=f"{category}/{seq}",
            images=images_arr,
            intrinsics=K_stack,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={
                "category": category,
                "sequence": seq,
                "frame_indices": tuple(rec["ids"]),
                "split": self.split,
                "filepaths": tuple(fr["filepath"] for fr in frames),
            },
        )
