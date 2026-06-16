"""NYU-v2 evaluation on the dust3r-lineage *prepared* set (CUT3R / MonST3R).

CUT3R Table 1 and MonST3R Table 3 evaluate NYU-v2 not on the
``nyu_depth_v2_labeled.mat`` set the :mod:`plumbline.datasets.nyuv2` loader
reads, but on the **MonST3R-prepared** set: the HuggingFace
``sayakpaul/nyu_depth_v2`` *val* split (654 ``.h5`` files) decoded to

    <root>/nyu_images/<id>.png    # RGB, (480, 640)
    <root>/nyu_depths/<id>.npy    # metric depth (meters), (480, 640) float32

by MonST3R's ``datasets_preprocess/prepare_nyuv2.py`` (``h5['rgb']`` ->
PNG, ``h5['depth']`` -> ``.npy``). CUT3R's ``eval/monodepth`` consumes
exactly this layout (``data/nyu-v2/val/{nyu_images,nyu_depths}``).

Why this loader exists, in one line: plumbline's CUT3R *inference* is
byte-identical to CUT3R's own (``scripts/_cut3r_nyu_input_diff.py``,
``max|Δ|=0``) and the eval recipe is the same (per-frame median, no clip,
valid ``gt>0``), so the only reason ``cut3r-nyuv2-lineage`` reads 0.0777
vs paper 0.086 is that it scores plumbline's ``labeled.mat`` 654-Eigen
set with *filled* GT instead of this prepared set's raw ``.h5`` depth.
Scoring CUT3R here is the apples-to-apples reproduction of Table 1.

The set is reusable across the lineage (MonST3R / DUSt3R NYU cells too).
Stage with ``scripts/stage_nyu_cut3r_eval.sh`` (public HF download, no auth).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from plumbline.datasets._common import DatasetNotAvailable, env_path
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.nyuv2 import NYUV2_INTRINSICS
from plumbline.datasets.registry import register_dataset


@register_dataset("nyu-cut3r-eval")
class NYUCut3rEvalDataset(Dataset):
    """NYU-v2 prepared eval set (MonST3R/CUT3R lineage): PNG images + .npy depth.

    Parameters
    ----------
    root
        Directory containing ``nyu_images/`` and ``nyu_depths/``. If omitted,
        falls back to ``$NYU_CUT3R_EVAL_ROOT``.
    max_gt_depth
        Optional GT upper bound (``valid = (0 < d) & (d < max_gt_depth)``).
        CUT3R's NYU eval uses ``max_depth=None`` (no cap), so the default
        ``None`` matches the paper; kept as a knob for parity with other cells.
    """

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        max_gt_depth: float | None = None,
    ) -> None:
        root_path = Path(root) if root else env_path("NYU_CUT3R_EVAL_ROOT")
        if root_path is None or not root_path.exists():
            raise DatasetNotAvailable(
                "NYU-v2 CUT3R/MonST3R prepared eval set not found. Set "
                "--data-root or $NYU_CUT3R_EVAL_ROOT to a directory containing "
                "nyu_images/<id>.png + nyu_depths/<id>.npy. Stage it (public, "
                "no auth) with scripts/stage_nyu_cut3r_eval.sh."
            )
        self.root = root_path
        self.img_dir = root_path / "nyu_images"
        self.depth_dir = root_path / "nyu_depths"
        if not self.img_dir.is_dir() or not self.depth_dir.is_dir():
            raise DatasetNotAvailable(
                f"Expected {self.img_dir} and {self.depth_dir}; one is missing. "
                "Stage with scripts/stage_nyu_cut3r_eval.sh."
            )
        self._ids = sorted(p.stem for p in self.img_dir.glob("*.png"))
        if not self._ids:
            raise DatasetNotAvailable(f"No <id>.png under {self.img_dir}.")
        self.max_gt_depth = float(max_gt_depth) if max_gt_depth is not None else None

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Sample]:
        K = np.array(
            [
                [NYUV2_INTRINSICS[0], 0.0, NYUV2_INTRINSICS[2]],
                [0.0, NYUV2_INTRINSICS[1], NYUV2_INTRINSICS[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        K_stack = K[None]
        E_eye = np.eye(4, dtype=np.float32)[None]

        for sid in self._ids:
            depth_path = self.depth_dir / f"{sid}.npy"
            if not depth_path.exists():
                raise DatasetNotAvailable(f"Missing depth for {sid}: {depth_path}")
            rgb = np.asarray(Image.open(self.img_dir / f"{sid}.png").convert("RGB"), dtype=np.uint8)
            depth = np.load(depth_path).astype(np.float32)
            valid = depth > 0.0
            if self.max_gt_depth is not None:
                valid &= depth < self.max_gt_depth
            yield Sample(
                sample_id=sid,
                images=rgb[None],
                intrinsics=K_stack,
                extrinsics_gt=E_eye,
                depth_gt=depth[None],
                depth_valid=valid[None],
                metadata={"dataset": "nyu-cut3r-eval"},
            )
