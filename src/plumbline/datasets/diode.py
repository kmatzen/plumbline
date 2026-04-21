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
    "DIODEMogeEvalLoader",
    "load_diode_depth_m",
    "load_diode_depth_mask",
    "load_moge_depth_png",
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


def load_moge_depth_png(path: Path) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Decode MoGe's log-encoded 16-bit depth.png into (depth_m, valid_mask).

    MoGe's `Ruicheng/monocular-geometry-evaluation` dataset re-encodes
    DIODE depth as a 16-bit PNG where pixel values map to log-spaced
    depth in meters:

        t      = (uint16_val - 1) / 65533
        depth  = near ** (1 - t) * far ** t

    with two sentinels:

        v = 0      → invalid (NaN)
        v = 65535  → unbounded / sky (+inf)

    `near` and `far` are stored as PNG text metadata. The valid mask is
    simply ``np.isfinite(depth)`` — this is what the paper's
    ``compute_metrics`` uses (see ``moge/test/dataloader.py``
    ``_process_instance``). Sky pixels and invalid pixels are both
    excluded from scoring.

    Returns ``(depth_m, valid)`` where ``depth_m`` has NaN/+inf for
    invalid/sky and ``valid = np.isfinite(depth_m)``.
    """
    from PIL import Image

    img = Image.open(path)
    if img.mode != "I;16":
        raise ValueError(
            f"expected 16-bit PNG (mode I;16) from {path}; got mode {img.mode!r}"
        )
    near = img.info.get("near")
    far = img.info.get("far")
    if near is None or far is None:
        raise ValueError(
            f"{path} missing PNG text metadata 'near'/'far' — is this the MoGe "
            "re-encoded depth.png from Ruicheng/monocular-geometry-evaluation?"
        )
    near_f = float(near)
    far_f = float(far)
    v = np.asarray(img, dtype=np.int32)  # int32 to avoid uint16 wraparound in math
    # t in [0, 1]; the 0 and 65535 sentinels are handled AFTER decoding
    # so we don't divide by 0 here.
    t = (v - 1).astype(np.float64) / 65533.0
    depth = (near_f ** (1.0 - t)) * (far_f ** t)
    depth = depth.astype(np.float32)
    depth[v == 0] = np.float32("nan")
    depth[v == 65535] = np.float32("inf")
    valid = np.isfinite(depth)
    return depth, valid


@register_dataset("diode-moge-eval")
class DIODEMogeEvalLoader(Dataset):
    """DIODE loader matching MoGe's evaluation pipeline.

    Reads from the preprocessed HF bundle
    ``Ruicheng/monocular-geometry-evaluation``'s ``DIODE.zip`` — NOT the
    raw DIODE devkit. The bundle re-encodes depth as log-spaced 16-bit
    PNG where sky pixels become ``+inf`` and invalid pixels become
    ``NaN``; the evaluation mask is simply ``isfinite(depth)`` with no
    additional clip. See ``load_moge_depth_png`` docstring for format
    details.

    Expected layout (pointed at by ``--data-root`` or
    ``$DIODE_MOGE_ROOT``)::

        <root>/DIODE/
          .index.txt                           # 770 sample paths, line-wise
          val/indoors/scene_*/scan_*/<id>/
              image.jpg, depth.png,
              segmentation.png, meta.json
          val/outdoor/scene_*/scan_*/<id>/...

    Stage via::

        hf download Ruicheng/monocular-geometry-evaluation \\
            DIODE.zip --repo-type dataset --local-dir /tmp/moge_dl
        unzip /tmp/moge_dl/DIODE.zip -d $DIODE_MOGE_ROOT

    Parameters
    ----------
    root
        Points at a directory containing the unzipped ``DIODE/`` tree.
        Falls back to ``$DIODE_MOGE_ROOT``.
    domain
        ``"indoors"``, ``"outdoor"``, or ``"both"``. ``"both"``
        preserves the ``.index.txt`` order (770 lines).
    """

    split: str = "val"

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        domain: str = "both",
    ) -> None:
        root_path = Path(root) if root else env_path("DIODE_MOGE_ROOT")
        if root_path is None or not (root_path / "DIODE").exists():
            raise DatasetNotAvailable(
                "DIODE MoGe-eval bundle not found. Set --data-root or "
                "$DIODE_MOGE_ROOT to a directory containing DIODE/.index.txt "
                "plus the unzipped DIODE/val/{indoors,outdoor}/... tree. "
                "Stage via: hf download Ruicheng/monocular-geometry-evaluation "
                "DIODE.zip --repo-type dataset --local-dir <tmp> && "
                "unzip <tmp>/DIODE.zip -d $DIODE_MOGE_ROOT"
            )
        if domain not in ("indoors", "outdoor", "both", "indoor", "outdoors"):
            raise ValueError(
                f"domain must be 'indoors' | 'outdoor' | 'both'; got {domain!r}"
            )
        # Normalize alias forms to the on-disk names.
        domain = {"indoor": "indoors", "outdoors": "outdoor"}.get(domain, domain)

        self.root = root_path
        self.domain = domain

        index_path = root_path / "DIODE" / ".index.txt"
        lines = [
            ln.strip()
            for ln in index_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if domain != "both":
            needle = f"val/{domain}/"
            lines = [ln for ln in lines if ln.startswith(needle)]
        self._records: list[dict[str, Any]] = [
            {"sample_path": ln, "sample_id": ln.replace("/", "_")} for ln in lines
        ]

    # -- iteration -------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield self._load_sample(rec)

    def __len__(self) -> int:
        return len(self._records)

    # -- per-sample ------------------------------------------------------

    def _load_sample(self, rec: dict[str, Any]) -> Sample:
        sample_root = self.root / "DIODE" / rec["sample_path"]
        image = read_rgb_uint8(sample_root / "image.jpg")
        images = image[None]
        assert_valid_image(images, name=f"diode_moge/{rec['sample_id']}/image")

        depth, valid = load_moge_depth_png(sample_root / "depth.png")
        # Depth array may contain NaN / +inf for invalid/sky pixels. Our
        # assert_valid_depth requires finite values, so replace the
        # sentinels with a dummy (1.0 m) — they will be excluded from the
        # metric anyway via `depth_valid`.
        depth_clean = np.where(valid, depth, np.float32(1.0))
        depth_gt = depth_clean[None]
        depth_valid = valid[None]

        # meta.json has normalized intrinsics (K_norm with cx=0.5, cy=0.5
        # as fractions of the image size). Denormalize to pixel units.
        import json
        meta = json.loads((sample_root / "meta.json").read_text())
        K_norm = np.asarray(meta["intrinsics"], dtype=np.float32)
        h, w, _ = image.shape
        K_pix = K_norm.copy()
        K_pix[0, :] *= w  # fx, 0, cx rows scaled by image width
        K_pix[1, :] *= h  # 0, fy, cy rows scaled by image height
        K_stack = K_pix[None]
        E_eye = np.eye(4, dtype=np.float32)[None]

        assert_valid_intrinsics(K_stack, name=f"diode_moge/{rec['sample_id']}/intrinsics")
        assert_valid_extrinsics(E_eye, name=f"diode_moge/{rec['sample_id']}/extrinsics")
        assert_valid_depth(depth_gt, name=f"diode_moge/{rec['sample_id']}/depth")

        # Parse domain from the sample path.
        domain = "indoors" if "/indoors/" in rec["sample_path"] else "outdoor"
        return Sample(
            sample_id=rec["sample_id"],
            images=images,
            intrinsics=K_stack,
            extrinsics_gt=E_eye,
            depth_gt=depth_gt,
            depth_valid=depth_valid,
            metadata={
                "domain": domain,
                "split": self.split,
                "source": "moge_hf_bundle",
                "sample_path": rec["sample_path"],
            },
        )


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
