# Architecture + extending plumbline

For library users (call plumbline from Python) and extenders (add a new
dataset / model / protocol / metric). To run an existing reproduction,
start at `README.md` + `REPRODUCTIONS.md`.

## 1. Vocabulary

| Concept | Lives in | Owns |
|---|---|---|
| **Sample** | `datasets/base.py::Sample` | One scene's images + intrinsics + extrinsics + optional depth/point cloud GT |
| **Prediction** | `models/base.py::Prediction` | One adapter's output: depth, intrinsics, extrinsics, point map, confidence |
| **Dataset** | `datasets/` | Iterable of Samples in canonical conventions |
| **Model** | `models/` | Adapter that wraps an upstream checkpoint and emits Prediction |
| **Protocol** | `protocols/*.yaml` | Named preset of dataset-prep + eval parameters |
| **Metric** | `metrics/` | Pure numpy fn: `(pred, gt, valid?) → float` |

**Reproduction**: a YAML under `reproductions/` that glues one model to
one dataset under one protocol with a paper target + tolerance.

**Runner** (`runner.py`): iterates a Dataset, calls `Model.predict()`
with caching, applies scale alignment, computes Metrics, aggregates.

**Conventions** (`conventions.py`): coordinate frames, units, array
shapes. **Coordinate conversion happens inside the loader, exactly
once, at load time** — never in the runner or a metric.

## 2. Using plumbline as a library

```python
from plumbline.reproduce import run_reproduction
result = run_reproduction("da-v2-small-nyuv2", output="/tmp/result.json")
print(result.primary_metric, result.observed, result.paper_match)
```

Or skip the YAML and call directly:

```python
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.runner import run

model = MODEL_REGISTRY["depth-anything-v2"](variant="small", device="cuda:0")
dataset = DATASET_REGISTRY["nyuv2"](split="test", apply_eigen_crop=True, depth_field="raw")
report = run(model=model, dataset=dataset, tasks=["mono_depth"],
             scale_alignment="scale_shift", depth_clip=(1e-3, 10.0),
             max_views=1)
print(report.aggregate_metrics)
```

**Output JSON** (from `plumbline reproduce -o out.json`) includes
`aggregate_metrics`, `per_sample`, `config_hash`, `environment`. Read
`aggregate_metrics[primary_metric]` for the paper-match check.

**Prediction cache** at
`~/.cache/plumbline/predictions/<model>/<config_hash>/<dataset>/<sample_id>.npz`.
Re-run with the same model+dataset+config reuses cached predictions and
recomputes metrics from scratch — so changing alignment / clip / metric
does NOT require re-inferring. Clear with
`rm -rf ~/.cache/plumbline/predictions/<model>/<config_hash>` when the
adapter's output format changes.

## 3. Extending: Datasets

```python
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset

@register_dataset("mydataset")
class MyDataset(Dataset):
    name = "mydataset"
    split = "val"

    def __init__(self, *, root=None):
        self.root = root or os.environ.get("MYDATASET_ROOT")
        self._records = [...]   # manifest-cache the scan

    def __iter__(self):
        for rec in self._records:
            yield Sample(
                sample_id=rec["id"],
                images=load_rgb_uint8(rec["image"]),    # (N, H, W, 3) uint8 sRGB
                intrinsics=load_K_pixels(rec["meta"]),  # (N, 3, 3) input-image pixel frame
                extrinsics_gt=load_pose(rec["meta"]),   # (N, 4, 4) world_from_camera
                depth_gt=load_depth_meters(rec["depth"]),  # (N, H, W) float32 meters, NaN/0 invalid
                depth_valid=load_depth_mask(rec["depth"]),
            )
```

**Conventions a loader must honour:**

- Images `(N, H, W, 3)` uint8 sRGB. Never BGR.
- Depth `(N, H, W)` float32 meters; NaN or 0 = invalid.
- Intrinsics in input-image pixel frame (re-scale focal/principal if
  the loader resized).
- Extrinsics `world_from_camera`. For multi-view, first camera is the
  world origin — use `conventions.rebase_to_first_camera` if the source
  is in a different global frame.
- `point_cloud_gt` (optional) `(M, 3)` float32, world frame, same unit
  as `depth_gt`.

**Rules the runner relies on:**

- Coordinate conversion inside the loader, once, at load time.
- Manifest-based scan: cache a JSON manifest at
  `<root>/.plumbline_manifest/<name>_<split>_<kwargs>.jsonl` (see
  `_common.save_manifest / load_manifest`).
- `DatasetNotAvailable`: raise when the root is missing, with a message
  pointing at the expected layout + download URL.
- Stable sample IDs (deterministic strings — they're cache keys).

References: `nyuv2.py` (single .mat), `kitti.py` (drives + crop masks),
`eth3d.py` (multi-view + GT point cloud), `diode.py` (alternate-format
loader pair).

## 4. Extending: Models

```python
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
        min_views=1, max_views=math.inf,
        requires_intrinsics=False,
        default_resolution=(518, 518),
    )

    def __init__(self, *, device="cuda:0", variant="base"):
        self.device = device
        self.variant = variant
        self._model = None

    def _load(self):
        if self._model is not None: return
        torch = ensure_torch()
        # construct, load weights, .to(self.device).eval()
        self._model = ...

    def predict(self, images, intrinsics=None) -> Prediction:
        assert_valid_image(images, name="my-model/input")
        self._load()
        depth = ...  # (N, H, W) float32, input-image resolution
        assert_valid_depth(depth, name="my-model/output")
        return Prediction(depth=depth, metadata={
            "variant": self.variant, "native_space": "disparity",
            "alignment_hint": "scale_shift",
        })

    def config_hash(self) -> str:
        s = f"{self.name}@{self.version}/variant={self.variant}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]
```

**Conventions a Prediction must honour:**

- Depth in input-image pixel frame at input-image resolution. If the
  net ran at a different resolution, the adapter upsamples/unscales.
- Depth units: meters if `is_metric=True`, else dimensionless.
  Inverse-depth / disparity outputs convert to depth before returning
  (with a floor to avoid div-by-zero).
- `alignment_hint` in metadata tells the runner which `scale_alignment`
  mode is expected by default. Options: `none`, `scale_shift` (LSQ in
  disparity), `scale_shift_robust` (IRLS-Huber in disparity),
  `scale_shift_depth` (LSQ in depth), `scale_shift_clamped` (per-sample
  disparity floor), `median`. See `metrics/alignment.py`.
- `config_hash()` is the prediction-cache key. Include every knob that
  affects prediction output (variant, input_size, seed, dtype). Don't
  include `device`.

**Rules the runner relies on:**

- Lazy torch import via `_torch_utils.ensure_torch()`. The adapter
  module must import cleanly without torch installed.
- Lazy weight load in `_load()`, not `__init__`.
- Deterministic output: seed per-sample (`seed + sample_index`) for
  diffusion models so cache hits reproduce.
- Registration-on-import — `cli.py` eager-imports all adapters.

**Upstream-clone adapters** (mast3r, geowizard, cut3r): require the user
to `git clone` and set an env var (`$MAST3R_ROOT` etc.); in `_load()`
prepend that path to `sys.path`. For upstream code drifted against
current diffusers/transformers, monkey-patch in a shim (see
`GeoWizardAdapter._shim_diffusers_for_geowizard`).

## 5. Extending: Protocols

```yaml
# protocols/my_paper.yaml
name: my_paper
description: What paper+table+row this protocol encodes.
reference:
  paper: "Author et al. 2024, Title (arXiv:2401.XXXXX)"
  split_source: "upstream repo file or paper Section/Table"
fixed:
  dataset:
    name: nyuv2
    split: test
    kwargs: { apply_eigen_crop: true, depth_field: raw }
  depth_clip: [0.001, 10.0]
  max_views: 1
  tasks: [mono_depth]
  # Optional: scale_alignment, mask_boundaries, pointcloud_alignment,
  # aggregation, scene_voxel_size
```

Reproductions declare `protocol: my_paper` and inherit the `fixed`
block. Overriding a fixed field raises `ProtocolConflictError`. Create
a new protocol when the paper uses a materially different eval (sample
list, crop, depth clip, alignment solver, evaluation frame). Reuse when
only the model / alignment / tolerance differs.

## 6. Extending: Metrics

```python
def my_metric(pred, gt, valid=None) -> float:
    """One-line description. Cite paper + equation if not obvious."""
    mask = valid if valid is not None else np.ones_like(gt, dtype=bool)
    mask = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    return float(np.abs(pred[mask] - gt[mask]).mean() / gt[mask].mean())
```

Pure numpy. No I/O, no torch, no mutation. Inputs in canonical frame
(meters / world frame / `world_from_camera`). Return floats — never
NaN on a non-empty valid mask. Re-entrant.

A new task-level metric is a runner change — touch
`runner._per_sample_metrics` to route inputs and `aggregate_metrics` to
pool per-sample results.

## 7. Where everything plugs in

```
reproductions/my_row.yaml
  ↓  apply_protocol()  ← merges protocols/<name>.yaml fixed block
  ↓
run_reproduction()
  ↓  DATASET_REGISTRY[name](**kwargs) → Dataset
  ↓  MODEL_REGISTRY[name](**kwargs)   → Model
  ↓
runner.run()
  ↓  for each Sample:
  ↓     prediction = model.predict(sample.images, sample.intrinsics)
  ↓                   [cache hit? return cached .npz]
  ↓     pred_aligned = scale_alignment(prediction.depth, sample.depth_gt)
  ↓     sample_metrics = depth_metrics(pred_aligned, sample.depth_gt, valid)
  ↓  aggregate over per-sample metrics
  ↓
Report → output.json + stdout markdown
```

## 8. Non-obvious invariants

If you violate one of these, things silently go wrong:

- **`depth_valid` is load-bearing.** Metrics only count pixels where
  `depth_valid=True`. If a loader sets it to `None`, the runner derives
  via `conventions.depth_is_valid` (non-NaN, positive).
- **`config_hash` is a cache key**, not a logging aid. Every kwarg that
  affects prediction output (input size, seed, dtype, variant) must be
  in the hash, or stale cache entries will be reused silently.
- **Cache fingerprint covers input tensors but not GT tensors** (D21
  fix). If a loader changes `depth_gt` / `extrinsics_gt` shape or units
  while `images` stays the same, predictions cached under the old GT
  will silently score against new GT. When refactoring a loader's GT
  output, bump `config_hash` or `rm -rf` the prediction shard.
- **Scale alignment is applied BEFORE metrics, AFTER depth_clip.**
  Order: predict → align to GT scale → clip to `[min, max]` m →
  metric.
- **Protocol-fixed fields.** A protocol's `fixed:` block is binding;
  the merger raises on override.
- **`source_confidence: verified_pdf`** is a contract with the paper
  PDF, not a default. A reproduction without a direct-PDF-verified
  paper value uses `source_confidence: approximate` (or `null`) and
  stays out of the verified queue.

## 9. Sample-list determinism

Every reproduction must pin its sample selection in-repo so two hosts
agree on which samples were evaluated. Acceptable mechanisms:

1. Loader has a hard-coded default and the YAML accepts it (NYU
   `_nyuv2_eigen_test.txt`, DTU `DTU_MVS_TEST_SCANS`).
2. YAML references a sample-list file committed inside the repo
   (`reproductions/<name>.samples.txt`).
3. YAML lists `scenes: [...]` for SCENE-FILTER datasets where iteration
   order is deterministic given (scene, views_per_sample) — ETH3D's
   loader satisfies this.

No new reproduction may depend on a file under `$<DATASET>_ROOT` for
sample selection.

## 10. Known traps

Read before adding adapters or closing off-paper cells:

- **Extrinsic conventions.** Half the field uses `camera_from_world`, half
  `world_from_camera`. Every loader and adapter is an inversion risk — use
  assertion helpers.
- **Depth vs inverse depth.** Adapters must document native output and convert
  explicitly (disparity / 1/z / meters).
- **Resize interpolation.** Bilinear for RGB; **nearest** for depth GT and masks.
- **Focal length after resize.** Unscale predicted intrinsics back to input pixels.
- **ScanNet poses.** Filter `inf` extrinsics in the loader.
- **Sintel intrinsics.** Verify against the dataset README; do not assume a default focal.
- **VGGT non-determinism.** Some CUDA ops vary with seed; document tolerance, not bitwise equality.
- **Prediction cache.** Cache keys omit loader preprocessing — bump cache or delete shards after loader changes (`DISCREPANCIES.md` D21).
