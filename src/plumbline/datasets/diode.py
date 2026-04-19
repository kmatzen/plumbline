"""DIODE dataset loader.

DIODE (Dense Indoor/Outdoor DEpth; Vasiljevic et al. 2019,
arXiv:1908.00463) is a high-resolution RGB-D dataset captured with a
FARO Focus3D X330 laser scanner + FLIR Blackfly camera. Each sample is
a single 1024x768 RGB frame with a dense float32 depth map in meters
plus a boolean validity mask. Modern mono-depth papers (DA-V2,
Metric3Dv2, DA3, MoGe, Depth Pro) all report DIODE numbers, usually on
the val split (``val_indoor`` + ``val_outdoor``).

Expected layout (point ``--data-root`` or ``$DIODE_ROOT`` at this)::

    <root>/
      val/
        indoors/                               # note: plural
          scene_00019/
            scan_00183/
              00019_00183_indoors_110_000.png
              00019_00183_indoors_110_000_depth.npy
              00019_00183_indoors_110_000_depth_mask.npy
        outdoor/                               # note: singular
          scene_00022/
            scan_.../...
      train/
        indoors/ outdoor/ ...

Download (public, no ToS): https://diode-dataset.org. Fetch the val
archives first (``val.tar.gz``, ~2 GB); the training split is much
larger (~80 GB) and rarely used for evaluation.

Conventions
-----------
- RGB is (768, 1024, 3) uint8 sRGB.
- Depth is (768, 1024) float32 meters. DIODE's .npy files ship as
  (H, W, 1); we squeeze the trailing axis on load.
- Depth mask is (768, 1024) uint8; 1 = valid LiDAR return, 0 = invalid.
  The loader converts to bool and populates ``Sample.depth_valid``.
- Intrinsics: DIODE ships per-scan calibration JSON in some releases
  but the widely-cited approximation is fx = fy = 886.81, cx = 512,
  cy = 384 (used by the DIODE devkit's demo). This loader uses those
  as the default; pass ``intrinsic=(fx, fy, cx, cy)`` to override.
  For mono-depth evaluation with scale alignment the exact focal
  length is not load-bearing — it only affects metric-scale
  reconstructions.
- DIODE is a single-view benchmark; extrinsics are identity.
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
    "DIODE_INTRINSIC",
    "DIODEDataset",
    "load_diode_depth_m",
    "load_diode_depth_mask",
]

# DIODE devkit's demo intrinsic. Per-scan calibration may differ by a
# few pixels; the devkit itself uses these values for visualization.
DIODE_INTRINSIC: tuple[float, float, float, float] = (886.81, 886.81, 512.0, 384.0)

# DIODE's two domain subdirectories, exactly as they ship. The mismatch
# (plural "indoors", singular "outdoor") is a quirk of the archive layout
# — we treat both as valid tokens and let the loader map between user
# input and on-disk name.
_DOMAIN_DIRS: dict[str, str] = {
    "indoors": "indoors",
    "outdoor": "outdoor",
    # Friendly aliases that match the conventional English usage.
    "indoor": "indoors",
    "outdoors": "outdoor",
}


@register_dataset("diode")
class DIODEDataset(Dataset):
    """DIODE dataset loader.

    Parameters
    ----------
    root
        Dataset root containing ``<split>/<domain>/scene_*/scan_*/...``.
        If omitted, falls back to ``$DIODE_ROOT``.
    split
        ``"val"`` (default) or ``"train"``. DIODE does not publish a
        test split with GT.
    domain
        ``"indoors"``, ``"outdoor"``, or ``"both"`` (concatenates the
        two domains in a stable order). Aliases ``"indoor"`` and
        ``"outdoors"`` are accepted.
    intrinsic
        Optional ``(fx, fy, cx, cy)`` override. Defaults to
        :data:`DIODE_INTRINSIC`.
    scenes
        Optional scene whitelist (e.g. ``["scene_00019"]``).
    """

    split: str = "val"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        split: str = "val",
        domain: str = "indoors",
        intrinsic: tuple[float, float, float, float] | None = None,
        scenes: list[str] | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("DIODE_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "DIODE not found. Set --data-root or $DIODE_ROOT to a directory "
                "containing <val|train>/<indoors|outdoor>/scene_*/scan_*/*.png plus "
                "matching *_depth.npy and *_depth_mask.npy files. "
                "Download (public, no account): https://diode-dataset.org."
            )
        if split not in ("val", "train"):
            raise ValueError(f"DIODE split '{split}' unsupported; use 'val' or 'train'")

        if domain == "both":
            domain_dirs = ["indoors", "outdoor"]
        else:
            if domain not in _DOMAIN_DIRS:
                raise ValueError(
                    f"domain must be one of {[*sorted(_DOMAIN_DIRS), 'both']}; got {domain!r}"
                )
            domain_dirs = [_DOMAIN_DIRS[domain]]

        self.root = root_path
        self.split = split
        self.domain = domain
        self.intrinsic = intrinsic or DIODE_INTRINSIC
        self._K: NDArray[np.float32] = _intrinsic_matrix(self.intrinsic)

        scenes_tag = "all" if scenes is None else f"scenes{len(scenes)}"
        manifest_path = (
            self.root / ".plumbline_manifest" / f"diode_{split}_{domain}_{scenes_tag}.jsonl"
        )
        if manifest_path.exists():
            records = load_manifest(manifest_path)
        else:
            records = list(self._scan(domain_dirs, scenes))
            save_manifest(manifest_path, records)
        if scenes:
            records = [r for r in records if r["scene"] in scenes]
        self._records = records

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- scanning --------------------------------------------------------

    def _scan(self, domain_dirs: list[str], scenes: list[str] | None) -> Iterator[dict[str, Any]]:
        split_root = self.root / self.split
        if not split_root.exists():
            raise DatasetNotAvailable(f"Expected {split_root}; not found.")
        single_domain = len(domain_dirs) == 1
        for domain_name in domain_dirs:
            domain_root = split_root / domain_name
            if not domain_root.exists():
                if single_domain:
                    # The user asked for exactly this domain and it's missing;
                    # fail loudly instead of silently iterating 0 samples.
                    raise DatasetNotAvailable(
                        f"DIODE domain tree not found at {domain_root}. "
                        "Download the corresponding archive from https://diode-dataset.org."
                    )
                # domain="both" mode: a partial download is fine; move on.
                continue
            scene_dirs = sorted(p for p in domain_root.iterdir() if p.is_dir())
            if scenes is not None:
                wanted = set(scenes)
                scene_dirs = [p for p in scene_dirs if p.name in wanted]
            for scene_dir in scene_dirs:
                scan_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir())
                for scan_dir in scan_dirs:
                    # Anchor the scan on *_depth.npy — the RGB .png alone isn't
                    # a unique indicator because the scan directory also holds
                    # *_normal.npy and (in some releases) per-scan JSON files.
                    for depth_path in sorted(scan_dir.glob("*_depth.npy")):
                        base = depth_path.name[: -len("_depth.npy")]
                        rgb_path = scan_dir / f"{base}.png"
                        mask_path = scan_dir / f"{base}_depth_mask.npy"
                        if not rgb_path.exists() or not mask_path.exists():
                            continue
                        yield {
                            "sample_id": f"{domain_name}/{scene_dir.name}/{scan_dir.name}/{base}",
                            "domain": domain_name,
                            "scene": scene_dir.name,
                            "scan": scan_dir.name,
                            "base": base,
                            "rgb_path": str(rgb_path.relative_to(self.root)),
                            "depth_path": str(depth_path.relative_to(self.root)),
                            "mask_path": str(mask_path.relative_to(self.root)),
                        }

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        image = read_rgb_uint8(self.root / rec["rgb_path"])
        images = image[None]  # (1, H, W, 3)
        assert_valid_image(images, name=f"diode/{rec['sample_id']}/image")

        depth = load_diode_depth_m(self.root / rec["depth_path"])
        depth_gt = depth[None]  # (1, H, W)

        mask = load_diode_depth_mask(self.root / rec["mask_path"])
        depth_valid = mask[None]  # (1, H, W) bool

        K_stack = self._K[None]  # (1, 3, 3)
        E_eye = np.eye(4, dtype=np.float32)[None]  # (1, 4, 4)

        assert_valid_intrinsics(K_stack, name=f"diode/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"diode/{rec['sample_id']}/extrinsics")
        assert_valid_depth(depth_gt, name=f"diode/{rec['sample_id']}/depth")

        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "domain": rec["domain"],
                "scene": rec["scene"],
                "scan": rec["scan"],
                "split": self.split,
                "intrinsic_source": (
                    "user-supplied" if self.intrinsic != DIODE_INTRINSIC else "diode_devkit_default"
                ),
            },
        )


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------


def load_diode_depth_m(path: Path) -> NDArray[np.float32]:
    """Load a DIODE depth .npy as a float32 meters array.

    DIODE ships depth as ``(H, W, 1)`` float32; we squeeze the trailing axis.
    Invalid pixels are conveyed via a separate mask file
    (:func:`load_diode_depth_mask`), not a sentinel depth value, so we do
    not zero out any pixels here.
    """
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D depth from {path}; got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_diode_depth_mask(path: Path) -> NDArray[np.bool_]:
    """Load a DIODE depth-mask .npy as a boolean validity array.

    DIODE masks are uint8 with ``1 = valid LiDAR return`` and ``0 =
    invalid``. We convert to bool so it can slot straight into
    ``Sample.depth_valid``.
    """
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask from {path}; got shape {arr.shape}")
    return np.ascontiguousarray(arr.astype(bool))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intrinsic_matrix(
    fxycxy: tuple[float, float, float, float],
) -> NDArray[np.float32]:
    fx, fy, cx, cy = fxycxy
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
