# Architecture + extending plumbline

This document is for two audiences:

1. **Library users** who want to call plumbline from Python (not via the
   CLI) — evaluate a model on a dataset, cache predictions, parse the
   output JSON.
2. **Extenders** who want to add a new **dataset**, **method** (model
   adapter), **protocol**, or **metric** to the harness.

If you are trying to run an existing reproduction, start at the top-level
`README.md` + `REPRODUCTIONS.md` instead.

---

## 1. Vocabulary

Five abstractions you will see throughout the codebase:

| Concept | What it is | Owns | Lives in |
|---|---|---|---|
| **Sample** | One scene = ``N`` views of RGB + GT | Images, intrinsics, extrinsics, optional depth / point cloud | `src/plumbline/datasets/base.py::Sample` |
| **Prediction** | One adapter's output on one Sample | Depth, intrinsics, extrinsics, point map, confidence | `src/plumbline/models/base.py::Prediction` |
| **Dataset** | Iterable of Samples in canonical conventions | File I/O, coord conversion, manifest caching | `src/plumbline/datasets/` |
| **Model** | Wraps an upstream checkpoint + predicts Prediction from images | Device/dtype/resize/normalization, convention conversion | `src/plumbline/models/` |
| **Protocol** | Named preset of dataset-prep + eval parameters | Sample enumeration, crop, depth clip, task list | `protocols/*.yaml` |
| **Metric** | Pure numpy function from prediction + GT to float | No I/O, no side effects | `src/plumbline/metrics/` |

**Reproduction**: a YAML under `reproductions/` that glues one model to
one dataset under one protocol, declares a paper target + tolerance,
and is the acceptance test that a given combo reproduces a published
number.

**Runner**: the glue at `src/plumbline/runner.py` that iterates a
Dataset, calls a Model on each Sample (with prediction caching), applies
scale alignment, computes the Metrics, and aggregates.

**Conventions** (`src/plumbline/conventions.py`): the shared contract for
coordinate frames, units, and array shapes. **Coordinate conversion
happens inside the loader, exactly once, at load time** — never in the
runner or a metric.

---

## 2. Using plumbline as a library

The CLI is the fastest path, but every piece is importable.

### 2.1 Run a reproduction from Python

```python
from plumbline.reproduce import run_reproduction

result = run_reproduction(
    "da-v2-small-nyuv2",        # name of a reproductions/*.yaml
    output="/tmp/result.json",  # optional — same JSON the CLI writes
)
print(result.primary_metric, result.observed, result.paper_match)
```

### 2.2 Run a model on a dataset without a YAML

```python
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.runner import run as run_runner

model_cls = MODEL_REGISTRY["depth-anything-v2"]
model = model_cls(variant="small", device="cuda:0")

ds_cls = DATASET_REGISTRY["nyuv2"]
dataset = ds_cls(split="test", apply_eigen_crop=True, depth_field="raw")

report = run_runner(
    model=model,
    dataset=dataset,
    tasks=["mono_depth"],
    scale_alignment="scale_shift",
    depth_clip=(1e-3, 10.0),
    max_views=1,
)
print(report.aggregate_metrics)
```

### 2.3 Call a model directly

```python
import numpy as np
from plumbline.models.registry import MODEL_REGISTRY

model = MODEL_REGISTRY["depth-anything-v2"](variant="small")
images = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
prediction = model.predict(images)
print(prediction.depth.shape)  # (1, 480, 640)
```

### 2.4 Parse the output JSON

`plumbline reproduce <name> -o out.json` writes a JSON with:

```json
{
  "schema_version": "0.1",
  "model": "...", "model_version": "...",
  "dataset": "...", "split": "...",
  "tasks": ["mono_depth"],
  "scale_alignment": "scale_shift",
  "n_total": 654, "n_evaluated": 654, "n_skipped": 0,
  "aggregate_metrics": {"abs_rel": 0.051, "rmse": 0.244, "delta_1": 0.97, ...},
  "per_sample": [ {"sample_id": "nyuv2_00000", "metrics": {...}, ...}, ...],
  "per_scene_metrics": { /* populated only when aggregation="scene" */ },
  "config_hash": "fd701fa1e30cbe54",
  "environment": { "plumbline_version": "...", "torch_version": "...", ... }
}
```

Any downstream analysis should read `aggregate_metrics[<primary_metric>]`
for the paper-match check, and `per_sample` for distribution / outlier
analysis.

### 2.5 Prediction cache

The runner caches model outputs at
`~/.cache/plumbline/predictions/<model>/<config_hash>/<dataset>/<sample_id>.npz`.
A rerun with the same model+dataset+config reuses the cached prediction
(and still recomputes metrics from scratch — so changing the alignment,
depth clip, or metric does NOT require re-inferring). Clear with
`rm -rf ~/.cache/plumbline/predictions/<model>/<config_hash>` when
you've changed the adapter's output format.

---

## 3. Extending: Datasets

A Dataset iterates `Sample` objects already in canonical conventions.

### 3.1 Minimum recipe

```python
# src/plumbline/datasets/mydataset.py
from collections.abc import Iterator
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

@register_dataset("mydataset")
class MyDataset(Dataset):
    name = "mydataset"
    split = "val"

    def __init__(self, *, root=None):
        self.root = root or os.environ.get("MYDATASET_ROOT")
        # ...manifest-cache the list of sample paths here
        self._records = [...]

    def __iter__(self) -> Iterator[Sample]:
        for rec in self._records:
            yield Sample(
                sample_id=rec["id"],
                images=load_rgb_uint8(rec["image"]),
                intrinsics=load_K_pixels(rec["meta"]),
                extrinsics_gt=load_pose_world_from_camera(rec["meta"]),
                depth_gt=load_depth_meters(rec["depth"]),
                depth_valid=load_depth_mask(rec["depth"]),
                metadata={"scene": rec["scene"]},
            )

    def __len__(self) -> int:
        return len(self._records)
```

### 3.2 Conventions a loader must honour

- **Images**: `(N, H, W, 3)` uint8 sRGB. Never BGR.
- **Depth**: `(N, H, W)` float32 meters. NaN or 0 for invalid.
- **Intrinsics**: `(N, 3, 3)` float32 pixel-space — in the **input-image
  coordinate frame**, so if the loader did any resize it must re-scale
  the focal/principal to match.
- **Extrinsics**: `(N, 4, 4)` float32 `world_from_camera`. For
  multi-view, first camera is the world origin — use
  `plumbline.conventions.rebase_to_first_camera` if the source is in a
  different global frame.
- **point_cloud_gt** (optional): `(M, 3)` float32, **world frame**, same
  unit as `depth_gt`. Needed for Chamfer / F-score metrics.

### 3.3 Rules the runner relies on

- **Coordinate conversion happens inside the loader, once, at load
  time.** The runner never touches coords; metrics operate on the
  conventional frame.
- **Manifest-based scan**: if the loader's `__init__` costs O(samples)
  on disk, cache a JSON manifest at
  `<root>/.plumbline_manifest/<name>_<split>_<kwargs>.jsonl`. See
  `plumbline.datasets._common.save_manifest / load_manifest`. Iteration
  then reads from the manifest, not from a directory scan.
- **DatasetNotAvailable**: raise this when the root is missing, with a
  message pointing at the expected layout + download URL.
- **Stable sample IDs**: deterministic, serializable strings. They're
  prediction-cache keys; a rename invalidates the cache.
- **`subset(n)` method** (recommended): deterministic first-N sample
  selection for quick dev runs.

### 3.4 Tests

- Add a synthetic-fixture test in `tests/test_datasets.py`: create a
  temp dir with 1–2 fake samples, instantiate, iterate, assert shapes +
  dtypes + conventions.

### 3.5 Real examples

- `src/plumbline/datasets/nyuv2.py` — single-file .mat; no manifest.
- `src/plumbline/datasets/kitti.py` — scan of many drives + crop masks.
- `src/plumbline/datasets/eth3d.py` — multi-view + GT point cloud.
- `src/plumbline/datasets/diode.py` — two loaders
  (`DIODEDataset` + `DIODEMogeEvalLoader`) sharing helpers; good
  reference for alternate-format cases.

---

## 4. Extending: Methods (Models)

A Model is an adapter that wraps an upstream checkpoint and converts its
output into a canonical `Prediction`.

### 4.1 Minimum recipe

```python
# src/plumbline/models/mymodel.py
import math, hashlib
import numpy as np
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model
from plumbline.models._torch_utils import ensure_torch
from plumbline.conventions import assert_valid_depth, assert_valid_image

@register_model("my-model")
class MyModelAdapter(Model):
    version = "1.0"
    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=False,
        min_views=1,
        max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(518, 518),
    )

    def __init__(self, *, device="cuda:0", variant="base"):
        self.device = device
        self.variant = variant
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        torch = ensure_torch()
        # ...construct upstream model, load weights, .to(self.device).eval()
        self._model = ...

    def predict(self, images, intrinsics=None) -> Prediction:
        assert_valid_image(images, name="my-model/input")
        self._load()
        # ...preprocess to the network's expected resolution/normalisation,
        # run inference, post-process to canonical units + conventions.
        depth = ...  # (N, H, W) float32
        assert_valid_depth(depth, name="my-model/output")
        return Prediction(
            depth=depth,
            metadata={"variant": self.variant, "native_space": "disparity",
                      "alignment_hint": "scale_shift"},
        )

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
```

### 4.2 Conventions a Prediction must honour

- **Depth in the input-image pixel frame** at the input-image
  resolution. If the adapter ran the net at some other resolution, it
  must upsample/unscale the output back to `(H, W)` of the input.
- **Depth units**: meters if `is_metric=True`, else dimensionless
  relative depth. Inverse-depth / disparity outputs must be converted
  to depth before returning (with a floor to avoid div-by-zero).
- **`alignment_hint` in `metadata`** tells the runner which
  `scale_alignment` mode is expected by default. Options currently:
  `"none"`, `"scale_shift"` (LSQ in disparity), `"scale_shift_robust"`
  (IRLS-Huber in disparity), `"scale_shift_depth"` (LSQ in depth
  space), `"median"` (ratio-of-medians). See
  `src/plumbline/metrics/alignment.py`.
- **`config_hash()`** is the prediction-cache key. Include every knob
  that affects prediction output (variant, input_size, seed, dtype,
  ...). Don't include `device` — same weights on different hardware
  should hash identically.

### 4.3 Rules the runner relies on

- **Lazy torch import** via `plumbline.models._torch_utils.ensure_torch()`.
  The adapter module must import cleanly without torch installed, so
  `plumbline list-models` works on environments that only have a subset
  of extras.
- **Lazy weight load**: don't load weights in `__init__`; do it in
  `_load()` on first `predict()`. Tests instantiate adapters without
  GPU.
- **Deterministic output**: when the model has random sampling (e.g.
  diffusion), seed per-sample (`seed + sample_index`) so cache hits
  reproduce.
- **Registration-on-import**: just importing the module should register
  the adapter. Eager-import all adapters in `src/plumbline/cli.py` so
  `plumbline list-models` shows everything.
- **No torch in the runner's eyes**: the adapter owns resolution,
  normalization, device, and convention conversion. The runner
  receives a `Prediction` of numpy arrays.

### 4.4 Upstream repo / pip-package adapters

Several real adapters wrap upstream repos that don't ship on PyPI:
`mast3r`, `geowizard`, `pi3`, `depth-anything-v2` (paper `.pth`). The
pattern is:

1. Require the user to `git clone` the upstream repo and set an env var
   (e.g. `$GEOWIZARD_ROOT`, `$DAV2_ROOT`).
2. In `_load()`, prepend that path to `sys.path` so upstream imports
   resolve.
3. For upstream code that's drifted against current diffusers /
   transformers, monkey-patch the namespaces at import time in a shim
   (see `GeoWizardAdapter::_shim_diffusers_for_geowizard`).

### 4.5 Tests

- `tests/test_model_adapters.py`: smoke — registration, instantiation
  without GPU, `config_hash` determinism, view-bound enforcement.
- `tests/test_model_weights.py` (`@pytest.mark.weights`): exercises the
  full load + predict path; skipped without the `models` extra.

### 4.6 Real examples

- `src/plumbline/models/depth_anything_v2.py` — HF `AutoModel` +
  upstream `.pth` loader, two source paths selectable via `source=`.
- `src/plumbline/models/marigold.py` — diffusers pipeline, ensemble,
  seeded per sample.
- `src/plumbline/models/metric3d_v2.py` — `torch.hub.load` + external
  `mmengine` / `mmcv-lite` deps.
- `src/plumbline/models/vggt.py` — multi-view transformer emitting
  depth + point map + extrinsics.

---

## 5. Extending: Protocols

A **protocol** is a YAML under `protocols/` that pins the
dataset-preparation + evaluation parameters a specific paper's row uses.

### 5.1 Minimum recipe

```yaml
# protocols/my_paper.yaml
name: my_paper
description: >
  What paper+table+row this protocol encodes, in one sentence.

reference:
  paper: "Author et al. 2024, Title (arXiv:2401.XXXXX)"
  split_source: "upstream repo file or paper Section/Table"

fixed:
  dataset:
    name: nyuv2          # a registered dataset
    split: test
    kwargs:
      apply_eigen_crop: true
      depth_field: raw
  depth_clip: [0.001, 10.0]   # null for no clip
  max_views: 1
  tasks: [mono_depth]
  # Optional: scale_alignment, mask_boundaries, pointcloud_alignment,
  # aggregation, scene_voxel_size, etc.
```

Reproduction YAMLs then declare `protocol: my_paper` and inherit the
`fixed` block. If a reproduction tries to override a fixed field to a
different value, `apply_protocol()` raises `ProtocolConflictError`.
This is how the harness enforces "every row that cites paper X uses
exactly paper X's protocol."

### 5.2 When to create a new protocol vs reuse

**Create a new one** when the paper you're citing uses a materially
different eval than any existing protocol: different sample list, crop,
depth clip, alignment solver, or evaluation frame. For example,
`kitti_eigen_garg` (Monodepth2 652 frames + Garg crop + [1e-3, 80] m
clip) and `kitti_eigen_crop` (Eigen crop, otherwise identical) are
different protocols because the crop materially affects AbsRel.

**Reuse** when the only per-reproduction difference is the model /
alignment / tolerance — those stay in the reproduction YAML.

### 5.3 Real examples

- `protocols/nyu_eigen_2014.yaml` — the most-used protocol, Silberman
  1449 restricted by Eigen `testNdxs` (654).
- `protocols/kitti_eigen_garg.yaml` — the 652-frame Monodepth2
  `eigen_benchmark` split, Uhrig dense GT, Garg crop, [1e-3, 80] clip.
- `protocols/diode_moge.yaml` — MoGe paper's DIODE val protocol,
  pointed at the `diode-moge-eval` loader + no depth clip.
- `protocols/dtu_vggt_table2.yaml` — MVS dense-point-cloud with ICP
  alignment + scene-level aggregation.

---

## 6. Extending: Metrics

Metrics are **pure numpy** functions: `(pred, gt, valid?) → float | dict[str, float]`.
No I/O. No torch. No mutation of inputs.

### 6.1 Minimum recipe

```python
# src/plumbline/metrics/mymetric.py
import numpy as np
from numpy.typing import NDArray

def my_metric(
    pred: NDArray[np.float32],
    gt: NDArray[np.float32],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    """One-line description. Cite the paper + equation if not obvious.

    Convention: pred and gt are in the same unit + frame; valid is a
    boolean mask selecting pixels to include.
    """
    mask = valid if valid is not None else np.ones_like(gt, dtype=bool)
    mask = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    return float(np.abs(pred[mask] - gt[mask]).mean() / gt[mask].mean())
```

Then add an `__all__` entry + re-export in `src/plumbline/metrics/__init__.py`
if the metric is meant to be public.

### 6.2 Conventions

- **Inputs in canonical frame**. Metrics trust the caller: depth in
  meters, point clouds in world frame, poses `world_from_camera`.
- **Broadcast-safe**. Shape conventions match the runner's arrays
  (`(N, H, W)` for depth, `(M, 3)` for point clouds).
- **Return floats**. Never NaN on a non-empty valid mask — if the
  metric is undefined for a given valid set, return the sentinel the
  paper uses (usually +inf or 0) and document it.
- **No side effects**. A metric is called O(samples) and sometimes
  again at aggregate time; it must be re-entrant.

### 6.3 Hooking into the runner

Metrics in `src/plumbline/metrics/{depth,pointmap,pose}.py` are called
directly from `src/plumbline/runner.py::_compute_task_metrics`. Adding a
new *task-level* metric (not just a per-depth variant) is a runner
change — touch `_per_sample_metrics` to route
`pred`, `gt`, and any task-specific auxiliary (like `point_cloud_gt`)
to your function, and `aggregate_metrics` to pool per-sample results.

### 6.4 Real examples

- `src/plumbline/metrics/depth.py::abs_rel`, `rmse`, `delta_threshold`.
- `src/plumbline/metrics/pointmap.py::chamfer_distance`,
  `accuracy_completeness`, `f_score`.
- `src/plumbline/metrics/pose.py::pairwise_pose_auc` —
  multi-view pose AUC, the VGGT / MASt3R / DUSt3R aggregation.
- `src/plumbline/metrics/alignment.py` — not metrics per se but the
  scale-alignment solvers (`scale_shift`, `scale_shift_robust`,
  `scale_shift_depth`, `median`) that sit upstream of depth metrics.

---

## 7. Where everything plugs in

Top-level flow of `plumbline reproduce my-row`:

```
reproductions/my_row.yaml
  ↓  apply_protocol()  ← merges protocols/<name>.yaml fixed block
  ↓
run_reproduction()
  ↓  DATASET_REGISTRY[name](**kwargs) → Dataset
  ↓  MODEL_REGISTRY[name](**kwargs)   → Model
  ↓
runner.run()  [src/plumbline/runner.py]
  ↓  for each Sample:
  ↓     prediction = model.predict(sample.images, sample.intrinsics)
  ↓                   [cache hit? return cached .npz]
  ↓     pred_aligned = scale_alignment(prediction.depth, sample.depth_gt)
  ↓     sample_metrics = depth_metrics(pred_aligned, sample.depth_gt, valid)
  ↓  aggregate_metrics = mean/agg over all per-sample metrics
  ↓
Report  →  output.json + stdout markdown
```

Every box above is a registry lookup or a conventional-interface call.
No adapter or dataset knows about any specific other — you extend one
at a time.

---

## 8. Non-obvious invariants

A short list of "if you violate this, things silently go wrong":

- **`depth_valid` is load-bearing.** Metrics only count pixels where
  `depth_valid=True`. If a loader sets it to `None`, the runner
  derives it via `conventions.depth_is_valid` (non-NaN, positive).
- **`config_hash` is a cache key**, not a logging aid. If your adapter
  has a kwarg that changes the prediction (input size, seed, dtype,
  variant), it must be in the hash — otherwise stale cache entries
  will be reused silently.
- **Scale alignment is applied BEFORE metrics, AFTER depth_clip.** The
  order is: predict → align to GT scale → clip to protocol's
  `[min, max]` m → metric. Changing either step shifts numbers.
- **Protocol-fixed fields.** If a protocol pins `depth_clip: [1e-3,
  80]`, no reproduction may set a different clip. The merger raises
  `ProtocolConflictError`. Use this to prevent silent per-reproduction
  drift from the paper's published protocol.
- **`source_confidence: verified_pdf`** is a contract with the paper
  PDF, not a default. A reproduction without a direct-PDF-verified
  paper value should use `source_confidence: approximate` (or `null`)
  and stay out of the verified queue.
