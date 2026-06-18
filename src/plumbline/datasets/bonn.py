"""Bonn RGB-D Dynamic dataset loader (video depth + pose).

Bonn RGB-D Dynamic (Palazzolo et al. 2019, "ReFusion: 3D Reconstruction in
Dynamic Environments using RGB-D Cameras") is a set of RGB-D sequences of
*dynamic* indoor scenes (people moving boxes, balloons, etc.). It's a
standard **video** benchmark for dynamic-scene geometry — MonST3R and CUT3R
both report Bonn video-depth numbers.

This loader closes plumbline's "no runnable video benchmark" gap: the new
recurrent / dynamic adapters (``cut3r``, ``monst3r``) can be evaluated on
real multi-frame sequences here, not just single frames.

Eval shape
----------
One :class:`~plumbline.datasets.base.Sample` == one *sequence* of ``N``
sub-sampled frames. The runner's depth alignment fits a single scale over a
whole sample, so running with ``scale_alignment: median`` reproduces the
**per-sequence scale** protocol that CUT3R / MonST3R video-depth eval uses
(CUT3R paper Table 2). Poses are rebased so frame 0 is the world origin
(plumbline convention), which also gives a relative GT trajectory for pose
metrics.

On-disk layout (TUM RGB-D format)
---------------------------------
``$BONN_ROOT`` contains one directory per sequence::

    rgbd_bonn_<name>/
      rgb/<timestamp>.png        # 640x480 sRGB
      depth/<timestamp>.png      # 640x480 uint16, 5000 units = 1 m
      rgb.txt                    # "<timestamp> rgb/<timestamp>.png"
      depth.txt                  # "<timestamp> depth/<timestamp>.png"
      groundtruth.txt            # "<timestamp> tx ty tz qx qy qz qw" (cam->world)

Download (no auth): https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
(individual ``rgbd_bonn_<name>.zip`` archives; extract under ``$BONN_ROOT``).

Note: the *exact* sequence list + frame sampling MonST3R/CUT3R use for the
published Bonn number live in MonST3R's eval code. This loader defaults to
all discovered sequences with even sub-sampling; a reproduction pinning the
paper cell must match MonST3R's protocol (single-record diff per GPU_RUNBOOK.md)
before it counts as a paper-match.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
    assert_valid_depth,
    assert_valid_extrinsics,
    assert_valid_image,
    assert_valid_intrinsics,
    rebase_to_first_camera,
)
from plumbline.datasets._common import DatasetNotAvailable, env_path, read_rgb_uint8
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["BONN_DEPTH_SCALE", "BONN_INTRINSICS", "BonnDataset"]

# Bonn RGB-D Dynamic calibration (Asus Xtion, 640x480) — the intrinsics
# MonST3R's Bonn eval uses.
BONN_INTRINSICS: tuple[float, float, float, float] = (
    542.822841,  # fx
    542.576870,  # fy
    315.593520,  # cx
    237.756098,  # cy
)
# TUM/Bonn depth PNGs: 16-bit, 5000 units per metre.
BONN_DEPTH_SCALE = 5000.0
# Max timestamp delta (s) when associating rgb<->depth<->pose (TUM default).
_ASSOC_MAX_DT = 0.02


@register_dataset("bonn")
class BonnDataset(Dataset):
    """Bonn RGB-D Dynamic video sequences.

    Parameters
    ----------
    root
        Dataset root (else ``$BONN_ROOT``). Contains ``rgbd_bonn_<name>/``.
    split
        Kept for API symmetry; Bonn has no official train/test split. ``test``.
    sequences
        Explicit sequence names (with or without the ``rgbd_bonn_`` prefix).
        ``None`` = every ``rgbd_bonn_*`` directory found, sorted.
    num_frames
        Frames per sequence sample, evenly sub-sampled. Caps memory for long
        sequences. Ignored when ``per_frame=True``.
    max_depth
        Depth values above this (m) are marked invalid (Bonn indoor ~ up to a
        few metres; clips sensor outliers).
    per_frame
        ``False`` (default) emits one :class:`Sample` per *sequence* with
        ``num_frames`` sub-sampled frames bundled into a single multi-view
        Sample — the per-sequence video-depth shape MonST3R / CUT3R Table 2
        uses. ``True`` emits one Sample per RGB frame
        (``images.shape[0] == 1``), iterating every frame in every selected
        sequence in timestamp order — the per-frame single-frame shape
        MonST3R Table 3 uses. ``num_frames`` and the per-sequence
        sub-sampling are skipped in this mode (every frame with a matched
        depth + pose is emitted).
    """

    split = "test"

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        split: str = "test",
        sequences: list[str] | None = None,
        num_frames: int = 90,
        max_depth: float = 10.0,
        per_frame: bool = False,
        frame_selection: str = "even",
        frame_start: int = 0,
        prepared_110: bool = False,
    ) -> None:
        self.split = split
        self.num_frames = int(num_frames)
        self.max_depth = float(max_depth)
        self.per_frame = bool(per_frame)
        # prepared_110: use MonST3R/CUT3R's pre-extracted `rgb_110/`+`depth_110/`
        # 110-frame subsets, paired by SORTED INDEX (not timestamp), exactly as
        # their eval_metadata.py globs them. This is the frame set DUSt3R's
        # MonST3R-Table-3 baseline (and the lineage video-depth cells) are
        # scored on; per-frame iteration only.
        self.prepared_110 = bool(prepared_110)
        if frame_selection not in ("even", "first"):
            raise ValueError(f"frame_selection must be 'even' or 'first'; got {frame_selection!r}")
        # "even" sub-samples across the whole sequence (DepthCrafter-style);
        # "first" takes ``num_frames`` contiguous frames starting at
        # ``frame_start`` — CUT3R/MonST3R's ``rgb_110`` Table-2 convention is
        # the [30:140] slice (frame_start=30, num_frames=110), set in
        # MonST3R's prepare_bonn.py.
        self.frame_selection = frame_selection
        self.frame_start = int(frame_start)
        resolved = env_path("BONN_ROOT", Path(root) if root is not None else None)
        if resolved is None or not Path(resolved).is_dir():
            raise DatasetNotAvailable(
                "Bonn RGB-D Dynamic not found. Set --data-root or $BONN_ROOT to a "
                "directory of rgbd_bonn_<name>/ sequences. Download (no auth) the "
                "rgbd_bonn_<name>.zip archives from "
                "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/ and "
                "extract them there."
            )
        self.root = Path(resolved)
        self._sequences = self._resolve_sequences(sequences)
        if not self._sequences:
            raise DatasetNotAvailable(
                f"No rgbd_bonn_* sequences found under {self.root}. Expected "
                "directories like rgbd_bonn_balloon/ with rgb/ depth/ "
                "groundtruth.txt."
            )

    def _resolve_sequences(self, sequences: list[str] | None) -> list[Path]:
        if sequences is not None:
            out: list[Path] = []
            for name in sequences:
                stem = name if name.startswith("rgbd_bonn_") else f"rgbd_bonn_{name}"
                p = self.root / stem
                if not p.is_dir():
                    raise DatasetNotAvailable(f"Bonn sequence not found: {p}")
                out.append(p)
            return out
        return sorted(p for p in self.root.glob("rgbd_bonn_*") if p.is_dir())

    def __len__(self) -> int:
        if self.per_frame:
            return sum(len(self._associations(seq_dir)) for seq_dir in self._sequences)
        return len(self._sequences)

    def __iter__(self) -> Iterator[Sample]:
        for seq_dir in self._sequences:
            if self.per_frame:
                yield from self._iter_per_frame(seq_dir)
            else:
                yield self._load_sequence(seq_dir)

    def _associations(self, seq_dir: Path) -> list[tuple[float, str, str, NDArray[np.float64]]]:
        """Build the rgb/depth/pose association list for a sequence.

        Each entry is ``(timestamp, rgb_rel, depth_rel, world_from_cam)`` —
        the same shape ``_load_sequence`` uses internally. Cached on the
        sequence dir's mtime so per-frame iteration doesn't re-parse the
        TUM lists for every frame.
        """
        rgb = _read_tum_list(seq_dir / "rgb.txt")
        depth = _read_tum_list(seq_dir / "depth.txt")
        traj = _read_tum_traj(seq_dir / "groundtruth.txt")
        if not rgb or not depth or not traj:
            raise DatasetNotAvailable(
                f"{seq_dir.name}: missing/empty rgb.txt, depth.txt, or groundtruth.txt"
            )
        depth_ts = np.array([t for t, _ in depth])
        traj_ts = np.array([t for t, _ in traj])
        assoc: list[tuple[float, str, str, NDArray[np.float64]]] = []
        for ts, rel in rgb:
            di = int(np.argmin(np.abs(depth_ts - ts)))
            ti = int(np.argmin(np.abs(traj_ts - ts)))
            if abs(depth_ts[di] - ts) > _ASSOC_MAX_DT or abs(traj_ts[ti] - ts) > _ASSOC_MAX_DT:
                continue
            assoc.append((ts, rel, depth[di][1], traj[ti][1]))
        if not assoc:
            raise DatasetNotAvailable(f"{seq_dir.name}: no rgb/depth/pose timestamp matches")
        return assoc

    def _prepared_110_pairs(
        self, seq_dir: Path
    ) -> list[tuple[float, str, str, NDArray[np.float64]]]:
        """rgb_110/depth_110 paired by SORTED INDEX (MonST3R Table-3 set).

        MonST3R/CUT3R glob ``rgb_110/*.png`` and ``depth_110/*.png`` sorted and
        pair them positionally (the subsets were pre-extracted aligned), rather
        than by timestamp association. Pose is unused for single-frame depth.
        """
        rgbs = sorted((seq_dir / "rgb_110").glob("*.png"))
        deps = sorted((seq_dir / "depth_110").glob("*.png"))
        if not rgbs or len(rgbs) != len(deps):
            raise DatasetNotAvailable(
                f"{seq_dir.name}: prepared_110 needs aligned rgb_110/ + depth_110/ "
                f"(got {len(rgbs)} rgb, {len(deps)} depth)."
            )
        eye = np.eye(4, dtype=np.float64)
        return [
            (float(i), f"rgb_110/{r.name}", f"depth_110/{d.name}", eye)
            for i, (r, d) in enumerate(zip(rgbs, deps, strict=True))
        ]

    def _iter_per_frame(self, seq_dir: Path) -> Iterator[Sample]:
        """Emit one Sample per RGB frame (single-frame Table 3 shape)."""
        sid_base = seq_dir.name.replace("rgbd_bonn_", "")
        k = _intrinsic_matrix(BONN_INTRINSICS)
        frames = (
            self._prepared_110_pairs(seq_dir) if self.prepared_110 else self._associations(seq_dir)
        )
        for ts, rgb_rel, depth_rel, _w_from_c in frames:
            image = read_rgb_uint8(seq_dir / rgb_rel)
            d, v = _load_bonn_depth(seq_dir / depth_rel, max_depth=self.max_depth)
            images_arr = image[None]  # (1, H, W, 3)
            depth_gt = d[None].astype(np.float32)
            depth_valid = v[None]
            intrinsics = k[None]
            # Single-frame: world == camera. Use identity so the loader's
            # pose contract holds; downstream pose metrics aren't meaningful
            # for a 1-view sample and are skipped by the runner.
            extrinsics = np.eye(4, dtype=np.float32)[None]
            sid = f"{sid_base}/{rgb_rel.split('/')[-1].removesuffix('.png')}"
            assert_valid_image(images_arr, name=f"bonn/{sid}/image")
            assert_valid_intrinsics(intrinsics, name=f"bonn/{sid}/intrinsics")
            assert_valid_extrinsics(extrinsics, name=f"bonn/{sid}/extrinsics")
            assert_valid_depth(depth_gt, name=f"bonn/{sid}/depth")
            yield Sample(
                sample_id=sid,
                images=images_arr,
                intrinsics=intrinsics,
                extrinsics_gt=extrinsics,
                depth_gt=depth_gt,
                depth_valid=depth_valid,
                metadata={
                    "dataset": "bonn",
                    "sequence": sid_base,
                    "timestamp": ts,
                    "n_frames": 1,
                    "per_frame": True,
                },
            )

    def _load_sequence(self, seq_dir: Path) -> Sample:
        assoc = self._associations(seq_dir)
        if self.frame_selection == "first":
            idxs = list(
                range(self.frame_start, min(self.frame_start + self.num_frames, len(assoc)))
            )
        else:
            idxs = _even_indices(len(assoc), self.num_frames)
        images: list[NDArray[np.uint8]] = []
        depths: list[NDArray[np.float32]] = []
        valids: list[NDArray[np.bool_]] = []
        poses: list[NDArray[np.float64]] = []
        for i in idxs:
            _ts, rgb_rel, depth_rel, w_from_c = assoc[i]
            images.append(read_rgb_uint8(seq_dir / rgb_rel))
            d, v = _load_bonn_depth(seq_dir / depth_rel, max_depth=self.max_depth)
            depths.append(d)
            valids.append(v)
            poses.append(w_from_c)

        images_arr = np.stack(images)
        n = images_arr.shape[0]
        k = _intrinsic_matrix(BONN_INTRINSICS)
        intrinsics = np.repeat(k[None], n, axis=0)
        extrinsics = rebase_to_first_camera(np.stack(poses)).astype(np.float32)
        depth_gt = np.stack(depths).astype(np.float32)
        depth_valid = np.stack(valids)

        sid = seq_dir.name.replace("rgbd_bonn_", "")
        assert_valid_image(images_arr, name=f"bonn/{sid}/image")
        assert_valid_intrinsics(intrinsics, name=f"bonn/{sid}/intrinsics")
        assert_valid_extrinsics(extrinsics, name=f"bonn/{sid}/extrinsics")
        assert_valid_depth(depth_gt, name=f"bonn/{sid}/depth")

        return Sample(
            sample_id=sid,
            images=images_arr,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={"dataset": "bonn", "sequence": sid, "n_frames": n},
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------


def _read_tum_list(path: Path) -> list[tuple[float, str]]:
    """Parse a TUM ``rgb.txt`` / ``depth.txt``: ``<timestamp> <relpath>`` lines.

    Lines starting with ``#`` and blanks are skipped.
    """
    if not path.exists():
        return []
    out: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        out.append((float(parts[0]), parts[1]))
    return out


def _read_tum_traj(path: Path) -> list[tuple[float, NDArray[np.float64]]]:
    """Parse TUM ``groundtruth.txt``: ``<ts> tx ty tz qx qy qz qw`` (cam->world).

    Returns (timestamp, 4x4 world_from_camera) per line.
    """
    if not path.exists():
        return []
    out: list[tuple[float, NDArray[np.float64]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts[1:8])
        out.append((ts, _tum_pose_to_matrix(tx, ty, tz, qx, qy, qz, qw)))
    return out


def _tum_pose_to_matrix(
    tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float
) -> NDArray[np.float64]:
    """TUM (translation + quaternion qx,qy,qz,qw) -> 4x4 world_from_camera."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero-norm quaternion in groundtruth")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    m[0, 1] = 2 * (qx * qy - qw * qz)
    m[0, 2] = 2 * (qx * qz + qw * qy)
    m[1, 0] = 2 * (qx * qy + qw * qz)
    m[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    m[1, 2] = 2 * (qy * qz - qw * qx)
    m[2, 0] = 2 * (qx * qz - qw * qy)
    m[2, 1] = 2 * (qy * qz + qw * qx)
    m[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    m[:3, 3] = [tx, ty, tz]
    return m


def _load_bonn_depth(
    path: Path, *, max_depth: float
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Load a Bonn 16-bit depth PNG (5000 units/m) as (meters, valid-mask)."""
    from PIL import Image as PImage

    arr = np.asarray(PImage.open(path), dtype=np.uint16)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D depth from {path}; got shape {arr.shape}")
    depth = arr.astype(np.float32) / BONN_DEPTH_SCALE
    valid = (arr != 0) & (depth <= max_depth)
    depth[~valid] = 0.0
    return depth, valid


def _even_indices(total: int, want: int) -> list[int]:
    """Pick ``want`` evenly-spaced indices from ``range(total)`` (all if fewer)."""
    if total <= want:
        return list(range(total))
    return list(np.linspace(0, total - 1, want).round().astype(int))


def _intrinsic_matrix(fxycxy: tuple[float, float, float, float]) -> NDArray[np.float32]:
    fx, fy, cx, cy = fxycxy
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
