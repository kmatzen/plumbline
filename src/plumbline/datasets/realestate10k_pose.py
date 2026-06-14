"""RealEstate10K multi-view-pose eval loader (DUSt3R/MASt3R/VGGT protocol).

Serves RealEstate10K as the multi-view relative-pose benchmark reported by:

- VGGT (arXiv:2503.11651) Table 1, "Re10K (unseen)" column: AUC@30 = 0.853.
- MASt3R (arXiv:2406.09756, ECCV 2024) Table 3, RealEstate10K column,
  pairwise block (b): MASt3R mAA(30) = 0.764, DUSt3R = 0.612.

All three follow the PoseDiffusion protocol ("Following [104], ... 1.8K
video clips from the test set of RealEstate10k. Each sequence is 10 frames
long, we evaluate relative camera poses between all possible 45 pairs, not
using ground-truth focals" — MASt3R §4.3). So this is the SAME pose recipe
as the CO3Dv2 eval (``co3dv2-vggt-pose-eval`` / ``co3dv2_vggt_pose``
protocol): one Sample per clip = 10 frames, 45 unordered pairs, mAA(30) =
AUC of min(RRA@30, RTA@30). It mirrors that loader's Sample shape exactly.

DATA AVAILABILITY (read before queueing) — RealEstate10K is the notorious
"links rot" dataset: Google distributes only per-clip camera ``.txt`` files
(URL + per-frame intrinsics+pose); the RGB frames must be scraped from
YouTube (``yt-dlp``) and cut at each timestamp, and a large fraction of the
source videos are now offline. There is NO clean HF/zip bundle (unlike
CO3Dv2's per-category zips or the MoGe eval bundles). This loader therefore
parses the STABLE official ``.txt`` format but expects an already-prepared
on-disk frame layout; it has NOT been exercised against real data. The first
box run validates the layout, the pose convention (see below), and the clip
subset. Created GPU-free in the 2026-05-28 coverage pass (verified targets;
unrunnable until frames are staged).

Expected on-disk layout (one subdir per clip), under ``$REALESTATE10K_ROOT``::

    <root>/<clip_id>/<clip_id>.txt   # official RealEstate10K camera file
    <root>/<clip_id>/<timestamp>.png # extracted RGB frame per .txt line
                                     # (.jpg also accepted)

Official ``.txt`` format (one clip per file)::

    https://www.youtube.com/watch?v=<id>            # line 0 (URL, skipped)
    <timestamp> <fx> <fy> <cx> <cy> 0 0 <12 floats> # one line per frame

where ``timestamp`` is microseconds (and names the frame image), the
intrinsics ``fx fy cx cy`` are NORMALIZED by image width/height, and the 12
floats are the row-major 3x4 ``cam_from_world`` (world-to-camera) extrinsic.

Pose convention
---------------
RealEstate10K's 3x4 matrix is ``cam_from_world`` (world-to-camera), the same
sense as OpenCV's extrinsic and as VGGT's ``convert_pt3d_RT_to_opencv``
output. plumbline's :class:`Sample` ``extrinsics_gt`` is the INVERSE,
``world_from_camera`` (c2w), rebased to first-camera-as-world — identical to
the CO3Dv2 loader. We therefore lift each 3x4 to 4x4, :func:`invert_pose`,
then :func:`rebase_to_first_camera`. (A w2c-vs-c2w mixup would transpose all
relative rotations and tank mAA, so this is the first thing to confirm on a
real run.)
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
from plumbline.datasets._common import DatasetNotAvailable, env_path, read_rgb_uint8
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["RealEstate10KPoseEvalLoader"]


def _find_frame(clip_dir: Path, timestamp: int) -> Path | None:
    """Frames are named by their microsecond timestamp; accept png or jpg."""
    for ext in ("png", "jpg", "jpeg"):
        p = clip_dir / f"{timestamp}.{ext}"
        if p.exists():
            return p
    return None


def _parse_camera_txt(txt_path: Path) -> list[dict[str, Any]]:
    """Parse a RealEstate10K clip ``.txt`` into per-frame records.

    Returns a list of ``{"timestamp": int, "fxfycxcy": (4,), "w2c": (3, 4)}``
    in file order. The first (URL) line is skipped.
    """
    frames: list[dict[str, Any]] = []
    for raw in txt_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("http"):
            continue
        parts = line.split()
        if len(parts) < 19:  # ts + 4 intrinsics + 2 zeros + 12 pose = 19
            continue
        ts = int(parts[0])
        fxfycxcy = np.asarray(parts[1:5], dtype=np.float64)
        # parts[5:7] are the two trailing zeros; pose is the last 12 floats.
        w2c = np.asarray(parts[7:19], dtype=np.float64).reshape(3, 4)
        frames.append({"timestamp": ts, "fxfycxcy": fxfycxcy, "w2c": w2c})
    return frames


@register_dataset("realestate10k-pose-eval")
class RealEstate10KPoseEvalLoader(Dataset):
    """RealEstate10K multi-view pose eval — one 10-frame Sample per clip.

    Parameters
    ----------
    root
        Directory of ``<clip_id>/`` subdirs (camera ``.txt`` + frames).
        Falls back to ``$REALESTATE10K_ROOT``.
    clips
        Optional whitelist of clip ids to restrict to — set this to the
        canonical 1.8K RealEstate10K test split (DUSt3R/PoseDiffusion) so the
        cell matches the paper subset. ``None`` = every clip found on disk
        (which will NOT match the paper unless the dir already IS that split).
    num_frames
        Frames sampled per clip (paper: 10).
    seed
        RNG seed for the per-clip frame sample (default 0). Note: exact frame
        selection cannot bit-match any one paper (VGGT and MASt3R use
        different codebases); mAA over 45 pairs is fairly stable to this.
    downsample_long_side
        If set, each served frame is BICUBIC-resized so its longer side equals
        this many pixels (frames already smaller are left untouched). A
        diagnostic knob, NOT a paper recipe: pose GT is resolution-independent
        and the protocol uses no GT focals, so feeding deliberately softer
        frames isolates each model's sensitivity to input resolution — the
        prime suspect for the VGGT-RE10K pose gap (correspondence-based
        DUSt3R/MASt3R vs feed-forward VGGT). ``None`` = serve frames as-staged.
    lowpass_sigma
        If set, each served frame is Gaussian-blurred in place at NATIVE
        resolution (no resize) with this sigma in source pixels. The decisive
        companion to ``downsample_long_side``: the downsample sweep's gain is
        confounded by the bicubic-down/VGGT-up resize round-trip, so it cannot
        distinguish "removing harmful high-frequency content helps" from "the
        288px scale itself matters". This knob isolates pure spatial-frequency
        content — same pixel count, aspect, and VGGT resize path as native,
        only the high-frequency band is attenuated. If AUC recovers toward the
        downsample-sweep peak under a native-res low-pass, the gap is
        artifact/HF-driven and the fix is an anti-alias prefilter. Diagnostic
        only, NOT a paper recipe. ``None`` = no blur.
    """

    split: str = "test"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        clips: list[str] | None = None,
        num_frames: int = 10,
        seed: int = 0,
        downsample_long_side: int | None = None,
        lowpass_sigma: float | None = None,
    ) -> None:
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2 for pose eval; got {num_frames}")
        if downsample_long_side is not None and downsample_long_side < 14:
            raise ValueError(
                f"downsample_long_side must be >= 14 if set; got {downsample_long_side}"
            )
        if lowpass_sigma is not None and lowpass_sigma < 0:
            raise ValueError(f"lowpass_sigma must be >= 0 if set; got {lowpass_sigma}")
        root_path = Path(root) if root else env_path("REALESTATE10K_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "RealEstate10K not found. Set --data-root or $REALESTATE10K_ROOT "
                "to a directory of <clip_id>/ subdirs, each holding the official "
                "RealEstate10K camera .txt plus per-timestamp frame images "
                "(<timestamp>.png). The camera .txt files are from "
                "https://google.github.io/realestate10k/; the RGB frames must be "
                "scraped from YouTube (yt-dlp) at each timestamp — many source "
                "videos are now offline."
            )
        self.root = root_path
        self.num_frames = int(num_frames)
        self.seed = int(seed)
        self.downsample_long_side = (
            int(downsample_long_side) if downsample_long_side is not None else None
        )
        self.lowpass_sigma = float(lowpass_sigma) if lowpass_sigma is not None else None
        whitelist = set(clips) if clips is not None else None

        self._records: list[dict[str, Any]] = []
        for clip_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            clip_id = clip_dir.name
            if whitelist is not None and clip_id not in whitelist:
                continue
            txts = sorted(clip_dir.glob("*.txt"))
            if not txts:
                continue
            cam = _parse_camera_txt(txts[0])
            # Keep only frames whose RGB was actually extracted on disk.
            usable = [fr for fr in cam if _find_frame(clip_dir, fr["timestamp"]) is not None]
            if len(usable) < self.num_frames:
                continue
            self._records.append({"clip_id": clip_id, "dir": clip_dir, "frames": usable})

        if not self._records:
            raise DatasetNotAvailable(
                f"No usable RealEstate10K clips under {root_path} (each needs a "
                f"camera .txt and >= {self.num_frames} matching <timestamp>.png "
                f"frames). Did the YouTube frame extraction succeed?"
            )

    def __iter__(self) -> Iterator[Sample]:
        rng = np.random.default_rng(self.seed)
        for rec in self._records:
            yield self._load_sample(rec, rng)

    def __len__(self) -> int:
        return len(self._records)

    def _load_sample(self, rec: dict[str, Any], rng: np.random.Generator) -> Sample:
        clip_id: str = rec["clip_id"]
        clip_dir: Path = rec["dir"]
        frames: list[dict[str, Any]] = rec["frames"]

        # Sample num_frames (sorted so pair enumeration is order-stable).
        idx = np.sort(rng.choice(len(frames), size=self.num_frames, replace=False))
        chosen = [frames[i] for i in idx]

        images: list[NDArray[np.uint8]] = []
        Ks: list[NDArray[np.float64]] = []
        cam_from_world: list[NDArray[np.float64]] = []
        for fr in chosen:
            frame_path = _find_frame(clip_dir, fr["timestamp"])
            if frame_path is None:
                raise FileNotFoundError(f"frame {fr['timestamp']} missing in {clip_dir}")
            img = read_rgb_uint8(frame_path)
            if self.downsample_long_side is not None:
                h0, w0 = img.shape[:2]
                long_side = max(h0, w0)
                if long_side > self.downsample_long_side:
                    # Simulate a lower-resolution / softer source frame. Pose GT
                    # is resolution-independent and the protocol uses no GT
                    # focals, so this isolates each model's sensitivity to input
                    # resolution (the prime suspect for the VGGT-RE10K gap).
                    from PIL import Image as _PImage

                    scale = self.downsample_long_side / long_side
                    new_w = max(1, round(w0 * scale))
                    new_h = max(1, round(h0 * scale))
                    img = np.asarray(
                        _PImage.fromarray(img).resize((new_w, new_h), _PImage.Resampling.BICUBIC),
                        dtype=np.uint8,
                    )
            if self.lowpass_sigma is not None and self.lowpass_sigma > 0:
                # Attenuate the high-frequency band at native resolution (no
                # resize) to test whether the downsample sweep's gain is HF/
                # artifact removal rather than the resize round-trip or the
                # pixel-count change. Gaussian sigma is in source pixels.
                from PIL import Image as _PImage
                from PIL import ImageFilter as _PFilter

                img = np.asarray(
                    _PImage.fromarray(img).filter(_PFilter.GaussianBlur(radius=self.lowpass_sigma)),
                    dtype=np.uint8,
                )
            H, W = img.shape[:2]
            images.append(img)
            fx, fy, cx, cy = fr["fxfycxcy"]
            Ks.append(
                np.array(
                    [[fx * W, 0.0, cx * W], [0.0, fy * H, cy * H], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            )
            w2c4 = np.eye(4, dtype=np.float64)
            w2c4[:3, :4] = fr["w2c"]
            cam_from_world.append(w2c4)

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
        assert_valid_image(images_arr, name=f"realestate10k/{clip_id}/image")

        K_stack = np.stack(Ks, axis=0).astype(np.float32)
        # RE10K stores cam_from_world (w2c); plumbline wants world_from_camera
        # (c2w) = inverse, rebased to first-camera-as-world (CO3Dv2 convention).
        E = invert_pose(np.stack(cam_from_world, axis=0))
        extrinsics = rebase_to_first_camera(E).astype(np.float32)
        assert_valid_intrinsics(K_stack, name=f"realestate10k/{clip_id}/K")
        assert_valid_extrinsics(extrinsics, name=f"realestate10k/{clip_id}/E")

        return Sample(
            sample_id=f"realestate10k/{clip_id}",
            images=images_arr,
            intrinsics=K_stack,
            extrinsics_gt=extrinsics,
            depth_gt=None,
            metadata={
                "clip_id": clip_id,
                "split": self.split,
                "timestamps": tuple(int(fr["timestamp"]) for fr in chosen),
            },
        )
