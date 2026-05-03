"""Microsoft 7-Scenes dataset loader.

7-Scenes (Shotton et al. 2013) is a small RGB-D sequence dataset
captured with Kinect v1 across 7 indoor rooms. It shipped with the
paper "Scene Coordinate Regression Forests for Camera Relocalization
in RGB-D Images" and remains a standard relocalization benchmark.

NOTE on paper citations: an earlier version of this docstring claimed
"MASt3R (§4.2 and Table 5) reports on it" — that is incorrect. The
MASt3R paper (arXiv:2406.09756) has Tables 1-4 only, and "7-Scenes"
appears zero times. MASt3R's only public 7-Scenes eval (in
`naver/dust3r`'s `dust3r_visloc/datasets/sevenscenes.py`) is visual
localization (PnP against retrieved map images, % at cm/deg
thresholds) — not pairwise relative pose. Pairwise pose AUC on
7-Scenes does appear in follow-ups (VGGT, CUT3R, Spann3R, Fast3R,
S-VGGT, MonST3R) but each defines its own pair sampler — pin a
specific paper before promoting any reproduction to verified_pdf.

Each frame on disk::

    <scene>/<sequence>/frame-<6-digit>.color.png   # 640x480 sRGB
    <scene>/<sequence>/frame-<6-digit>.depth.png   # 640x480 uint16, 1000 units/m
    <scene>/<sequence>/frame-<6-digit>.pose.txt    # 4x4 world_from_camera

Pose convention
---------------
Microsoft's official README is explicit that pose.txt stores the
camera-to-world transform (plumbline calls this ``world_from_camera``).
No inversion in the loader.

Intrinsics
----------
Kinect v1 at 640x480 is conventionally modelled with a single shared
pinhole for the factory-registered color/depth stream:

    fx = fy = 585.0
    cx, cy = 320.0, 240.0

These are the canonical Microsoft 7-Scenes Kinect-v1 SIMPLE_PINHOLE
intrinsics (also what MASt3R's visual-localization eval consumes via
kapture metadata). Users can override via ``intrinsic=(fx, fy, cx, cy)``.

Test split
----------
The canonical test sequences per scene, per Shotton 2013 §3 / the
Microsoft Research download page:

    chess      : seq-03, seq-05
    fire       : seq-03, seq-04
    heads      : seq-01
    office     : seq-02, seq-06, seq-07, seq-09
    pumpkin    : seq-01, seq-07
    redkitchen : seq-03, seq-04, seq-06, seq-12, seq-14
    stairs     : seq-01, seq-04

Depth validity
--------------
Kinect v1 encodes invalid pixels as 65535 (0xFFFF); 0 also means
invalid after normalization. ``depth_valid`` masks both.

Download
--------
Scene archives (one zip per scene, each ~100 MB-1 GB) are at
https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
(public, no login required).
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

__all__ = ["SEVEN_SCENES_TEST_SEQUENCES", "SevenScenesDataset"]

# Kinect v1 factory-registered pinhole, 640x480 (7-Scenes native).
SEVEN_SCENES_INTRINSIC: tuple[float, float, float, float] = (585.0, 585.0, 320.0, 240.0)

# Depth PNG encoding: per-pixel value / 1000 = meters.
_DEPTH_UNITS_PER_M: float = 1000.0

# Sentinel in 7-Scenes depth PNGs for "no return".
_DEPTH_INVALID_SENTINEL: int = 65535

# Canonical test-sequence split (Shotton 2013 + MSR download page).
SEVEN_SCENES_TEST_SEQUENCES: dict[str, tuple[str, ...]] = {
    "chess": ("seq-03", "seq-05"),
    "fire": ("seq-03", "seq-04"),
    "heads": ("seq-01",),
    "office": ("seq-02", "seq-06", "seq-07", "seq-09"),
    "pumpkin": ("seq-01", "seq-07"),
    "redkitchen": ("seq-03", "seq-04", "seq-06", "seq-12", "seq-14"),
    "stairs": ("seq-01", "seq-04"),
}


@register_dataset("7scenes")
class SevenScenesDataset(Dataset):
    """7-Scenes RGB-D + pose loader, sliding-window N-view sampling.

    Each :class:`Sample` is an N-view window (``views_per_sample``,
    default 2 for pair-based pose eval). Windows slide over frames
    within a single sequence with a configurable stride.

    Parameters
    ----------
    root
        7-Scenes root directory (contains ``<scene>/<sequence>/...``).
        Falls back to ``$SEVEN_SCENES_ROOT``.
    split
        ``"test"`` (default, uses :data:`SEVEN_SCENES_TEST_SEQUENCES`)
        or ``"custom"`` when explicit ``scenes``/``sequences`` are given.
    scenes
        Optional whitelist of scene names. Defaults to all 7.
    sequences
        Optional mapping ``{scene: [seq, ...]}``. Overrides the split's
        default sequences. Use for single-sequence dev runs.
    views_per_sample
        N-view window size. 2 is typical for MASt3R-style two-view
        pose eval; higher values enable multi-view tests.
    stride
        Frame stride between window starts. Larger = fewer, farther-
        apart pairs; 1 = every consecutive pair. Default 10 (≈0.3 s
        at 30 fps).
    baseline
        Temporal gap between the first and last view in a window.
        Larger baseline = wider-baseline pairs (harder pose). Default
        10; MASt3R-era papers sweep across values.
    intrinsic
        Optional ``(fx, fy, cx, cy)`` override.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        scenes: list[str] | None = None,
        sequences: dict[str, list[str]] | None = None,
        views_per_sample: int = 2,
        stride: int = 10,
        baseline: int = 10,
        intrinsic: tuple[float, float, float, float] | None = None,
    ) -> None:
        if split not in ("test", "custom"):
            raise ValueError(f"7-Scenes split must be 'test' or 'custom'; got {split!r}")
        if views_per_sample < 1:
            raise ValueError(f"views_per_sample must be >= 1; got {views_per_sample}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride}")
        if baseline < 1:
            raise ValueError(f"baseline must be >= 1; got {baseline}")

        root_path = Path(root) if root else env_path("SEVEN_SCENES_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "7-Scenes not found. Set --data-root or $SEVEN_SCENES_ROOT to a "
                "directory containing <scene>/<sequence>/frame-*.color.png plus "
                "matching *.depth.png and *.pose.txt. Download (public, no login): "
                "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/"
            )

        # Resolve the (scene -> sequences) mapping.
        if sequences is not None:
            seq_map = {k: list(v) for k, v in sequences.items()}
        elif split == "test":
            seq_map = {k: list(v) for k, v in SEVEN_SCENES_TEST_SEQUENCES.items()}
        else:
            raise ValueError("split='custom' requires an explicit sequences= argument")
        if scenes is not None:
            wanted = set(scenes)
            seq_map = {k: v for k, v in seq_map.items() if k in wanted}
            if not seq_map:
                raise ValueError(
                    f"no known scenes in {scenes!r}; valid: {sorted(SEVEN_SCENES_TEST_SEQUENCES)}"
                )

        self.root = root_path
        self.split = split
        self.scenes_map = seq_map
        self.views_per_sample = int(views_per_sample)
        self.stride = int(stride)
        self.baseline = int(baseline)
        self.intrinsic = intrinsic or SEVEN_SCENES_INTRINSIC
        self._K: NDArray[np.float32] = _intrinsic_matrix(self.intrinsic)

        # Deterministic scenes_tag so custom whitelists get distinct manifests.
        scenes_tag = "_".join(f"{s}-{len(v)}seq" for s, v in sorted(seq_map.items()))
        manifest_path = (
            self.root
            / ".plumbline_manifest"
            / f"7scenes_{split}_v{views_per_sample}_s{stride}_b{baseline}_{scenes_tag}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan())
            save_manifest(manifest_path, records)
        self._records = records

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self) -> Iterator[dict[str, Any]]:
        """Enumerate (scene, sequence, [frame-ids]) windows."""
        for scene in sorted(self.scenes_map):
            scene_dir = self.root / scene
            if not scene_dir.exists():
                # Partial download: skip scenes the user hasn't fetched.
                continue
            for seq in sorted(self.scenes_map[scene]):
                seq_dir = scene_dir / seq
                if not seq_dir.exists():
                    continue
                # Discover frame ids by scanning *.color.png.
                color_paths = sorted(seq_dir.glob("frame-*.color.png"))
                frame_ids = [p.name.split(".")[0] for p in color_paths]
                # Windows: stride=N, width=baseline+1 when views_per_sample=2;
                # general case picks `views_per_sample` frames spaced by baseline.
                width = (self.views_per_sample - 1) * self.baseline + 1
                if len(frame_ids) < width:
                    continue
                for start in range(0, len(frame_ids) - width + 1, self.stride):
                    window = [
                        frame_ids[start + k * self.baseline]
                        for k in range(self.views_per_sample)
                    ]
                    first = window[0]
                    yield {
                        "sample_id": f"{scene}/{seq}/{first}_v{self.views_per_sample}_b{self.baseline}",
                        "scene": scene,
                        "sequence": seq,
                        "frame_ids": window,
                    }

    # -- per sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        scene = rec["scene"]
        seq = rec["sequence"]
        frame_ids: list[str] = list(rec["frame_ids"])

        images: list[NDArray[np.uint8]] = []
        depths: list[NDArray[np.float32]] = []
        valids: list[NDArray[np.bool_]] = []
        poses: list[NDArray[np.float64]] = []

        for fid in frame_ids:
            base = self.root / scene / seq / fid
            rgb = read_rgb_uint8(Path(str(base) + ".color.png"))
            depth, valid = load_seven_scenes_depth_m(Path(str(base) + ".depth.png"))
            pose = load_seven_scenes_pose(Path(str(base) + ".pose.txt"))
            images.append(rgb)
            depths.append(depth)
            valids.append(valid)
            poses.append(pose)

        # 7-Scenes within a sequence is a fixed-pose camera moving through a
        # rigid scene; images are all 640x480 so stacking is clean.
        images_arr = np.stack(images, axis=0)
        assert_valid_image(images_arr, name=f"7scenes/{rec['sample_id']}/image")

        K_stack = np.repeat(self._K[None], len(frame_ids), axis=0)
        assert_valid_intrinsics(K_stack, name=f"7scenes/{rec['sample_id']}/intrinsics")

        # Poses in 7-Scenes are already camera-to-world (==world_from_camera).
        world_from_camera = np.stack(poses, axis=0)
        extrinsics = rebase_to_first_camera(world_from_camera).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name=f"7scenes/{rec['sample_id']}/extrinsics")

        depth_gt = np.stack(depths, axis=0).astype(np.float32)
        depth_valid = np.stack(valids, axis=0)
        assert_valid_depth(depth_gt, name=f"7scenes/{rec['sample_id']}/depth")

        return Sample(
            sample_id=rec["sample_id"],
            images=images_arr,
            intrinsics=K_stack,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "scene": scene,
                "sequence": seq,
                "frame_ids": tuple(frame_ids),
                "split": self.split,
                "intrinsic_source": (
                    "user-supplied" if self.intrinsic != SEVEN_SCENES_INTRINSIC else "kinect_v1_default"
                ),
            },
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------


def load_seven_scenes_depth_m(path: Path) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Load a 7-Scenes depth PNG as (meters, valid-mask).

    7-Scenes encodes depth as a 16-bit PNG with 1000 units = 1 metre.
    Invalid pixels are marked 65535 (sensor saturation / no return);
    0 also means invalid after normalization. Returns a float32 depth
    array and a bool validity mask.
    """
    from PIL import Image as PImage

    arr = np.asarray(PImage.open(path), dtype=np.uint16)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D depth from {path}; got shape {arr.shape}")
    valid = (arr != _DEPTH_INVALID_SENTINEL) & (arr != 0)
    depth = arr.astype(np.float32) / _DEPTH_UNITS_PER_M
    # Zero out invalid entries so downstream consumers that don't honour
    # depth_valid still see a sentinel-free array.
    depth[~valid] = 0.0
    return depth, valid


def load_seven_scenes_pose(path: Path) -> NDArray[np.float64]:
    """Load a 7-Scenes pose.txt as a 4x4 float64 ``world_from_camera`` matrix.

    File format: 4 rows × 4 whitespace-separated floats. No header.
    """
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4 pose matrix, got shape {arr.shape}")
    # Sanity: bottom row should be [0, 0, 0, 1] within float noise.
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
        raise ValueError(f"{path}: pose matrix last row {arr[3]} is not [0,0,0,1]")
    return arr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intrinsic_matrix(fxycxy: tuple[float, float, float, float]) -> NDArray[np.float32]:
    fx, fy, cx, cy = fxycxy
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
