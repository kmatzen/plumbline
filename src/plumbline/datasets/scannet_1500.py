"""ScanNet-1500 pose benchmark loader.

A fixed set of 1500 image pairs from 15 ScanNet test scenes, selected by
Sarlin et al. for SuperGlue (2020) and now the de-facto two-view pose
estimation benchmark — VGGT Table 4, MASt3R paper §4.1, LoFTR, and every
recent matching-and-pose paper reports on this split.

The pairs + GT poses + intrinsics are pinned in a public TXT file; the
IMAGES live under the ScanNet test split (auth-gated ToS). This loader
accepts the TXT and resolves the image paths under ``$SCANNET_ROOT``.

Pair list source (pinned, no auth needed): the magicleap/SuperGlue repo
at ``assets/scannet_test_pairs_with_gt.txt``. Each line:

    scans_test/<scene>/sens/<frame_0>.color.jpg \\
    scans_test/<scene>/sens/<frame_1>.color.jpg \\
    <dummy> <dummy> \\
    <K_0 3x3 9 floats> <K_1 3x3 9 floats> \\
    <T_0_to_1 4x4 16 floats>

The 4x4 is the GT relative pose ``T_0_to_1 = cam1_from_cam0``. plumbline's
convention is world_from_camera with camera 0 as world origin, so we
use:
    E_0 = identity  (cam0 == world)
    E_1 = inv(T_0_to_1)  (world_from_cam1)

Expected layout (same ``SCANNET_ROOT`` as :class:`ScanNetDataset`)::

    <root>/scans_test/<scene>/sens/<frame>.color.jpg

Access: http://www.scan-net.org/ (ToS-gated, free for research).

This loader is validated end-to-end on synthetic fixtures; real-data
runs require the user's signed ScanNet email-download to land. Once
``scans_test/scene0707_00/sens/frame-000015.color.jpg`` resolves, the
loader is ready without further code changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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

__all__ = ["ScanNet1500Dataset", "parse_scannet_1500_pairs"]


@register_dataset("scannet-1500")
class ScanNet1500Dataset(Dataset):
    """ScanNet-1500 two-view pose benchmark loader.

    Each :class:`Sample` is a 2-view pair: images stacked (2, H, W, 3),
    per-view intrinsics (2, 3, 3), and GT extrinsics (2, 4, 4) with camera
    0 as world origin. Depth GT is not provided by this benchmark — it's
    pose-only — so ``depth_gt`` stays None.

    Parameters
    ----------
    root
        ScanNet test-split root (contains ``scans_test/``). Falls back to
        ``$SCANNET_ROOT``.
    pairs_file
        Path to the SuperGlue ``scannet_test_pairs_with_gt.txt`` file
        (or an equivalent in the same format). Loader ships no bundled
        copy — the user passes one from the SuperGlue repo.
    """

    split: str = "test_1500"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        pairs_file: Path | str,
    ) -> None:
        root_path = Path(root) if root else env_path("SCANNET_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "ScanNet not found. Set --data-root or $SCANNET_ROOT to the ScanNet "
                "test-split root (with scans_test/<scene>/sens/*.color.jpg). Request "
                "access at http://www.scan-net.org/. Pair list comes from "
                "magicleap/SuperGluePretrainedNetwork assets/scannet_test_pairs_with_gt.txt "
                "(no auth; public)."
            )
        pairs_path = Path(pairs_file)
        if not pairs_path.exists():
            raise DatasetNotAvailable(
                f"pairs_file not found: {pairs_path}. Download from "
                "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/assets/scannet_test_pairs_with_gt.txt"
            )
        self.root = root_path
        self.pairs_file = pairs_path
        self._records = list(parse_scannet_1500_pairs(pairs_path))

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, object]) -> Sample:
        img0_path = self.root / str(rec["image_0"])
        img1_path = self.root / str(rec["image_1"])
        if not (img0_path.exists() and img1_path.exists()):
            raise DatasetNotAvailable(
                f"Missing ScanNet images for pair {rec['pair_id']}: "
                f"{img0_path} / {img1_path}. Extract the ScanNet test "
                "archive (requires signed ToS) under $SCANNET_ROOT."
            )
        img0 = read_rgb_uint8(img0_path)
        img1 = read_rgb_uint8(img1_path)
        # SuperGlue pairs sometimes span frames of subtly different
        # resolutions within a scene (different .sens export settings).
        # Upstream ScanNet color frames are all 1296x968 or so, but
        # defend against mismatch by requiring equal shape per pair —
        # resizing to a common size belongs to the model adapter, not
        # the loader.
        if img0.shape != img1.shape:
            raise ValueError(
                f"pair {rec['pair_id']}: image shape mismatch "
                f"{img0.shape} vs {img1.shape}"
            )
        images = np.stack([img0, img1], axis=0)
        assert_valid_image(images, name=f"scannet1500/{rec['pair_id']}/image")

        K = np.stack(
            [np.asarray(rec["K_0"], dtype=np.float32), np.asarray(rec["K_1"], dtype=np.float32)],
            axis=0,
        )
        assert_valid_intrinsics(K, name=f"scannet1500/{rec['pair_id']}/intrinsics")

        # Pair file stores T_0_to_1 (cam1_from_cam0). Construct
        # world_from_camera extrinsics with cam0 as world origin:
        #   E_0 = identity, E_1 = inv(T_0_to_1)
        T_cam1_from_cam0 = np.asarray(rec["T_0_to_1"], dtype=np.float64).reshape(4, 4)
        world_from_cam0 = np.eye(4, dtype=np.float64)
        world_from_cam1 = invert_pose(T_cam1_from_cam0)
        extrinsics = rebase_to_first_camera(
            np.stack([world_from_cam0, world_from_cam1], axis=0)
        ).astype(np.float32)
        assert_valid_extrinsics(extrinsics, name=f"scannet1500/{rec['pair_id']}/extrinsics")

        return Sample(
            sample_id=str(rec["pair_id"]),
            images=images,
            intrinsics=K,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={
                "scene": str(rec["scene"]),
                "frame_0": str(rec["frame_0"]),
                "frame_1": str(rec["frame_1"]),
                "split": self.split,
            },
        )


def parse_scannet_1500_pairs(path: Path) -> Iterator[dict[str, object]]:
    """Parse SuperGlue's ``scannet_test_pairs_with_gt.txt`` line-by-line.

    Each non-empty line is:

        <image_0_path> <image_1_path> 0 0 \\
        <K_0 9 floats> <K_1 9 floats> \\
        <T_cam1_from_cam0 16 floats>

    Yields one dict per pair with all parsed fields. The ``0 0`` filler
    between image paths and intrinsics is the legacy SuperGlue "overlap
    bucket" marker; we ignore it.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 2 image paths + 2 fillers + 9 + 9 + 16 = 38 tokens.
            if len(parts) < 38:
                raise ValueError(
                    f"{path}:{line_no}: expected >= 38 tokens, got {len(parts)}"
                )
            img0, img1 = parts[0], parts[1]
            # parts[2], parts[3] are the legacy overlap buckets — skip.
            K0 = np.asarray([float(x) for x in parts[4:13]], dtype=np.float64).reshape(3, 3)
            K1 = np.asarray([float(x) for x in parts[13:22]], dtype=np.float64).reshape(3, 3)
            T = np.asarray([float(x) for x in parts[22:38]], dtype=np.float64).reshape(4, 4)
            scene = img0.split("/")[-3] if "/" in img0 else "unknown"
            yield {
                "pair_id": f"pair_{line_no:05d}_{scene}",
                "scene": scene,
                "image_0": img0,
                "image_1": img1,
                "frame_0": Path(img0).stem,
                "frame_1": Path(img1).stem,
                "K_0": K0,
                "K_1": K1,
                "T_0_to_1": T,
            }


def _fetch_pairs_file() -> NDArray:
    """Placeholder — the canonical source is external. Document the URL in
    the loader's DatasetNotAvailable message and leave retrieval to the
    user (one-time ~200KB download from the SuperGlue repo)."""
    raise NotImplementedError("pairs file is external; see SuperGlue repo")
