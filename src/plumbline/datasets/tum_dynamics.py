"""TUM-RGBD dynamics video-pose loader (MonST3R / DAGE Table 4 protocol).

Serves the TUM-RGBD ``freiburg3`` *dynamic* sequences as the camera-trajectory
benchmark reported by MonST3R Table 4 and **DAGE Table 4** (Ngo et al. 2026,
arXiv:2603.03744), "TUM-Dynamics" column. One :class:`Sample` is one sequence's
full 90-frame trajectory; ATE / RPE-RMSE are computed under Sim(3) (Umeyama)
alignment via ``evo`` — the same apparatus plumbline already runs for
``dage-sintel-pose`` / ``monst3r-sintel-pose`` (only the dataset differs).

Protocol (verbatim from MonST3R's ``datasets_preprocess/prepare_tum.py``, which
DAGE's ``evaluation/relpose`` reuses unchanged — DAGE ``metadata.py`` has
``tum: {seq_list: None, full_seq: True, traj_format: 'tum'}``):

1. Associate each ``rgb.txt`` timestamp to the nearest ``groundtruth.txt``
   timestamp (greedy, ``max_difference = 0.02 s``, no offset) — the canonical
   TUM ``associate.py`` algorithm.
2. Sort matches by timestamp, then take ``frames[::3][:90]`` — the first 90
   frames at temporal stride 3.
3. The trajectory is the matched ground-truth poses (TUM format:
   ``tx ty tz qx qy qz qw``), which are **camera-to-world** (``world_from_camera``)
   — the position/orientation of the color camera in the world frame.

The eight MonST3R/DAGE sequences (``download_tum_dynamics.sh``)::

    rgbd_dataset_freiburg3_sitting_static       rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_sitting_xyz          rgbd_dataset_freiburg3_walking_xyz
    rgbd_dataset_freiburg3_sitting_halfsphere   rgbd_dataset_freiburg3_walking_halfsphere
    rgbd_dataset_freiburg3_sitting_rpy          rgbd_dataset_freiburg3_walking_rpy

Expected on-disk layout (the raw extracted TUM ``.tgz`` archives, one dir per
sequence) under ``--data-root`` or ``$TUM_ROOT``::

    <root>/rgbd_dataset_freiburg3_<name>/
        rgb.txt                  # "<timestamp> rgb/<timestamp>.png" per line
        rgb/<timestamp>.png
        groundtruth.txt          # "<timestamp> tx ty tz qx qy qz qw" per line

Stage with ``scripts/stage_tum_dynamics.py`` (public, no ToS). The loader does
the associate + stride-3 + first-90 prep itself, so no preprocessing step is
needed — only the extracted archives.

Pose convention
---------------
TUM ground truth is ``world_from_camera`` (c2w): ``(tx, ty, tz)`` is the camera
position in the world frame and ``(qx, qy, qz, qw)`` its orientation. plumbline's
:class:`Sample` ``extrinsics_gt`` is ``world_from_camera`` rebased to
first-camera-as-world. We build the 4x4 from translation + quaternion directly,
then :func:`rebase_to_first_camera`. (Sim(3) alignment in the metric absorbs the
rebasing, so it matches MonST3R's evo eval regardless.)
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import (
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

__all__ = [
    "TUM_DYNAMIC_SEQUENCES",
    "TUMDynamicsDataset",
    "associate_tum",
    "select_tum_frames",
    "tum_pose_to_matrix",
]

# The eight freiburg3 dynamic sequences MonST3R / DAGE evaluate on
# (download_tum_dynamics.sh). full_seq=True in DAGE's metadata simply means
# "iterate whatever is staged"; this is the staged set.
TUM_DYNAMIC_SEQUENCES: tuple[str, ...] = (
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
)

# Freiburg-3 RGB camera intrinsics (TUM benchmark ROS default), 640x480. Pose
# eval is feed-forward + Sim(3)-aligned so these are not used by the metric, but
# Sample requires a valid K per frame.
_FR3_FX, _FR3_FY, _FR3_CX, _FR3_CY = 535.4, 539.2, 320.1, 247.6
_FRAME_STRIDE = 3
_NUM_FRAMES = 90
_ASSOC_MAX_DIFF = 0.02  # seconds, TUM associate.py default


def _read_tum_file_list(path: Path) -> dict[float, list[str]]:
    """Parse a TUM ``stamp d1 d2 ...`` file → ``{stamp: [d1, d2, ...]}``.

    Port of ``associate.read_file_list`` (the canonical TUM tool prepare_tum.py
    uses): drop comment/blank lines, split on whitespace/comma/tab.
    """
    text = path.read_text()
    out: dict[float, list[str]] = {}
    for line in text.replace(",", " ").replace("\t", " ").split("\n"):
        line = line.strip()
        if not line or line[0] == "#":
            continue
        parts = [v for v in line.split(" ") if v]
        if len(parts) > 1:
            out[float(parts[0])] = parts[1:]
    return out


def associate_tum(
    first: dict[float, list[str]],
    second: dict[float, list[str]],
    *,
    offset: float = 0.0,
    max_difference: float = _ASSOC_MAX_DIFF,
) -> list[tuple[float, float]]:
    """Greedy nearest-timestamp association (TUM ``associate.associate``).

    Returns matched ``(stamp_first, stamp_second)`` pairs, sorted by timestamp —
    bit-for-bit the same selection as MonST3R's prepare_tum.py.
    """
    first_keys = set(first.keys())
    second_keys = set(second.keys())
    potential = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential.sort()
    matches: list[tuple[float, float]] = []
    for _diff, a, b in potential:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def tum_pose_to_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> NDArray[np.float64]:
    """TUM ``(t, quaternion)`` → 4x4 ``world_from_camera`` (c2w).

    Quaternion order is ``(qx, qy, qz, qw)`` (TUM convention); normalized here
    for safety against logging round-off.
    """
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if n == 0.0:
        raise ValueError("zero-norm quaternion in TUM ground truth")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = (tx, ty, tz)
    return E


def select_tum_frames(
    rgb_txt: Path, gt_txt: Path
) -> tuple[list[str], list[list[str]]]:
    """Run MonST3R's prepare_tum.py selection on a sequence's rgb/gt files.

    Single source of truth for "which 90 frames + poses" — used both by the
    loader's manifest scan and by ``scripts/stage_tum_dynamics.py`` (so staging
    can extract just those frames from the archive).

    Returns ``(rgb_relpaths, gt_rows)`` where each ``gt_row`` is
    ``[stamp, tx, ty, tz, qx, qy, qz, qw]`` (strings), aligned to ``rgb_relpaths``.
    """
    rgb_list = _read_tum_file_list(rgb_txt)
    gt_list = _read_tum_file_list(gt_txt)
    matches = associate_tum(rgb_list, gt_list)
    rgb_rel = [rgb_list[a][0] for a, _b in matches]
    gt_rows = [[str(b), *gt_list[b]] for _a, b in matches]
    return rgb_rel[::_FRAME_STRIDE][:_NUM_FRAMES], gt_rows[::_FRAME_STRIDE][:_NUM_FRAMES]


@register_dataset("tum-dynamics")
class TUMDynamicsDataset(Dataset):
    """TUM-RGBD freiburg3 dynamic-sequence video-pose loader.

    One :class:`Sample` per sequence = its first-90-frames-at-stride-3
    trajectory (MonST3R / DAGE Table 4 "TUM-Dynamics"). Pose-only
    (``depth_gt`` is left ``None``).

    Parameters
    ----------
    root
        Directory holding the extracted ``rgbd_dataset_freiburg3_*`` archives.
        Falls back to ``$TUM_ROOT``.
    scenes
        Optional subset of sequence names. Default: all eight dynamic
        sequences present on disk (:data:`TUM_DYNAMIC_SEQUENCES`).
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "test",
        scenes: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("TUM_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "TUM-dynamics not found. Set --data-root or $TUM_ROOT to a directory "
                "containing the extracted rgbd_dataset_freiburg3_*/ archives "
                "(rgb.txt, rgb/, groundtruth.txt). Stage with "
                "scripts/stage_tum_dynamics.py (public, no ToS)."
            )
        self.root = root_path
        self.split = split
        self._wanted = list(scenes) if scenes else list(TUM_DYNAMIC_SEQUENCES)

        # Key the manifest on the set of sequences actually present, so staging
        # more sequences after a first (partial) scan invalidates the cache
        # rather than silently serving the old, smaller set.
        present = sorted(
            s for s in TUM_DYNAMIC_SEQUENCES if (self.root / s / "rgb.txt").exists()
        )
        tag = hashlib.sha256("|".join(present).encode()).hexdigest()[:12]
        manifest_path = self.root / ".plumbline_manifest" / f"tum_dynamics_pose_{tag}.jsonl"
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan())
            save_manifest(manifest_path, records)
        wanted = set(self._wanted)
        self._records = [r for r in records if r["scene"] in wanted]
        if not self._records:
            raise DatasetNotAvailable(
                f"No TUM-dynamics sequences found under {self.root} matching {sorted(wanted)}. "
                "Expected rgbd_dataset_freiburg3_*/ with rgb.txt + groundtruth.txt."
            )

    # -- scanning --------------------------------------------------------

    def _scan(self) -> Iterator[dict[str, Any]]:
        for seq in TUM_DYNAMIC_SEQUENCES:
            seq_dir = self.root / seq
            rgb_txt = seq_dir / "rgb.txt"
            gt_txt = seq_dir / "groundtruth.txt"
            if not (rgb_txt.exists() and gt_txt.exists()):
                continue
            strided_rgb, strided_gt = select_tum_frames(rgb_txt, gt_txt)
            if len(strided_rgb) < 3:
                continue
            yield {
                "sample_id": f"{seq}/full_s{_FRAME_STRIDE}",
                "scene": seq,
                "image_paths": [str((seq_dir / p).relative_to(self.root)) for p in strided_rgb],
                # each gt row: [stamp, tx, ty, tz, qx, qy, qz, qw]
                "gt_rows": strided_gt,
            }

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        images = np.stack([read_rgb_uint8(self.root / p) for p in rec["image_paths"]], axis=0)
        assert_valid_image(images, name=f"tum/{rec['sample_id']}/image")

        Es = []
        for row in rec["gt_rows"]:
            _stamp, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in row)
            Es.append(tum_pose_to_matrix(tx, ty, tz, qx, qy, qz, qw))
        extrinsics = rebase_to_first_camera(np.stack(Es)).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name=f"tum/{rec['sample_id']}/E")

        n = images.shape[0]
        K = np.array(
            [[_FR3_FX, 0.0, _FR3_CX], [0.0, _FR3_FY, _FR3_CY], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        intrinsics = np.broadcast_to(K, (n, 3, 3)).copy()
        assert_valid_intrinsics(intrinsics, name=f"tum/{rec['sample_id']}/K")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=intrinsics,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={"scene": rec["scene"], "dataset": "tum-dynamics", "n_frames": n},
        )
