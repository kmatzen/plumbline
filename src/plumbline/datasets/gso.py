"""GSO (Google Scanned Objects) loader via MoGe's preprocessed evaluation.

Google Scanned Objects is a synthetic-render dataset from Google's
robotics team — 1030 single-object scenes rendered at 512×512 with
pristine GT depth. MoGe Table 1/2 reports GSO as the "synthetic
clean-GT" slot. plumbline picks GSO as one of the Sintel-depth
substitutes after the 2026-04-19 pivot (Sintel remained auth-gated).

We load GSO from MoGe's preprocessed bundle on HuggingFace
(``Ruicheng/monocular-geometry-evaluation``) since the upstream
Google release doesn't ship rendered RGB + depth pairs directly —
MoGe rendered the 1030 objects and packaged them.

Expected layout::

    <root>/
      <object_name>/
        image.jpg      # 512x512 RGB
        depth.png      # 512x512 uint16, log-encoded with near/far PNG metadata
        meta.json      # {"intrinsics": [[fx/W, 0, cx/W], [0, fy/H, cy/H], [0, 0, 1]]}

Download (public, no auth):

    pip install huggingface-hub
    hf download Ruicheng/monocular-geometry-evaluation \\
        --repo-type dataset --include 'GSO*' --local-dir data/moge_eval
    cd data/moge_eval && unzip GSO.zip

Depth decoding (per ``moge/utils/io.py`` read_depth)::

    t = (raw_uint16 - 1) / 65533
    depth = near ** (1 - t) * far ** t    # log-interpolation
    # raw == 0  → NaN (invalid)
    # raw == 65535 → inf (beyond far plane)

``near``/``far`` are stored in the PNG ``info`` dict (pHYs/iTXt chunks).
GSO's intrinsics are normalized: multiply fx, cx by image width to get
pixel-space; fy, cy by height. Principal point is at (0.5, 0.5) in
normalized coords = image centre in pixels.

GSO is single-object, single-view; extrinsics are identity.
"""

from __future__ import annotations

import io
import json
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
)
from plumbline.datasets._common import (
    DatasetNotAvailable,
    env_path,
    read_rgb_uint8,
)
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

__all__ = ["GSODataset", "read_moge_depth_png"]


@register_dataset("gso")
class GSODataset(Dataset):
    """GSO loader via MoGe's preprocessed HuggingFace bundle.

    Parameters
    ----------
    root
        Directory containing ``<object_name>/{image.jpg, depth.png,
        meta.json}`` subdirs. Falls back to ``$GSO_ROOT``.
    objects
        Optional whitelist (object name substring match). ``None`` =
        all 1030 objects.
    """

    split: str = "val"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        objects: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("GSO_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "GSO not found. Set --data-root or $GSO_ROOT to a directory "
                "containing <object_name>/{image.jpg, depth.png, meta.json}. "
                "Download from Ruicheng/monocular-geometry-evaluation on "
                "HuggingFace: `hf download Ruicheng/monocular-geometry-evaluation "
                "--repo-type dataset --include 'GSO*' --local-dir <root>/..`."
            )
        self.root = root_path
        # Skip non-dir entries like .index.txt that HF bundles.
        all_objects = sorted(
            p.name for p in root_path.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
        if objects is not None:
            wanted = set(objects)
            self.object_names = [n for n in all_objects if n in wanted]
        else:
            self.object_names = all_objects
        if not self.object_names:
            raise DatasetNotAvailable(
                f"No GSO object subdirs found under {root_path}. Each "
                "subdir must contain meta.json."
            )

    def __iter__(self) -> Iterator[Sample]:
        for name in self.object_names:
            yield self._load_sample(name)

    def __len__(self) -> int:
        return len(self.object_names)

    def _load_sample(self, name: str) -> Sample:
        sample_dir = self.root / name
        img = read_rgb_uint8(sample_dir / "image.jpg")
        images = img[None]  # (1, H, W, 3)
        assert_valid_image(images, name=f"gso/{name}/image")

        H, W, _ = img.shape

        depth = read_moge_depth_png(sample_dir / "depth.png")
        if depth.shape != (H, W):
            raise ValueError(
                f"gso/{name}: depth {depth.shape} mismatches image {(H, W)}"
            )
        # Replace NaN/inf with 0 (plumbline's invalid marker) so metrics
        # treat them as invalid via depth>0.
        depth_valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(depth_valid, depth, 0.0).astype(np.float32)
        depth_gt = depth[None]

        with (sample_dir / "meta.json").open() as f:
            meta = json.load(f)
        K_norm = np.asarray(meta["intrinsics"], dtype=np.float64)
        # Normalized: fx/W, fy/H, cx/W, cy/H. Un-normalise to pixel K.
        K = K_norm.copy()
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H
        K_stack = K[None].astype(np.float32)
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"gso/{name}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"gso/{name}/extrinsics")
        assert_valid_depth(depth_gt, name=f"gso/{name}/depth")

        return Sample(
            sample_id=f"gso/{name}",
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid[None],
            metadata={"object": name, "split": self.split},
        )


def read_moge_depth_png(path: Path) -> NDArray[np.float32]:
    """Decode a MoGe-encoded depth PNG back to float32 depth values.

    MoGe stores depth as a 16-bit PNG with per-image ``near`` and ``far``
    embedded in the PNG ``info`` dict. Encoding rule (from
    ``moge/utils/io.py::read_depth``)::

        raw == 0       → NaN (invalid, no GT)
        raw == 65535   → inf (beyond far plane)
        else:
            t = (raw - 1) / 65533      # in [0, 1]
            depth = near ** (1 - t) * far ** t   # log-interpolation

    Returns a float32 ``(H, W)`` array. NaN and inf are preserved at
    the loader level; caller converts them to the plumbline invalid
    marker (0).
    """
    from PIL import Image

    data = Path(path).read_bytes()
    pil_image = Image.open(io.BytesIO(data))
    raw = np.array(pil_image)
    if raw.dtype != np.uint16:
        raise ValueError(
            f"expected uint16 depth PNG from {path}; got {raw.dtype}"
        )
    info = pil_image.info
    if "near" not in info or "far" not in info:
        raise ValueError(
            f"{path}: MoGe depth PNG missing 'near'/'far' in PNG info dict."
        )
    near = float(info["near"])
    far = float(info["far"])
    mask_nan = raw == 0
    mask_inf = raw == 65535
    t = (raw.astype(np.float32) - 1.0) / 65533.0
    depth = (near ** (1.0 - t) * far ** t).astype(np.float32)
    if "unit" in info:
        depth *= float(info["unit"])
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth
