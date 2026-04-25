# plumbline — Implementation Plan

A reproducible evaluation harness for 3D geometric foundation models. Think
`lm-evaluation-harness` but for models like VGGT, Depth Anything 3, MASt3R,
Metric3Dv2, Depth Pro, MoGe.

This document is the spec. Work through it section by section. Ask before
deviating from the architecture; feel free to deviate on implementation
details within each section.

> **Status:** v0.1 in development. Live state lives in
> [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) (paper-match matrix) and
> [`docs/DISCREPANCIES.md`](./docs/DISCREPANCIES.md) (open issues).
> [`GPU_RUNBOOK.md`](./GPU_RUNBOOK.md) is the bring-up + thrift
> bootstrap doc for both human and autonomous-agent operators. § 10
> below has the v0.2 roadmap.

---

## 0. Context for the agent

- Owner: Kevin Blackburn-Matzen (Adobe, formerly Facebook CompPhoto/XR). Deep
  3D vision background. Has shipped VGGT-era systems in product. Has strong
  opinions about coordinate systems and color; defer to him on conventions.
- The goal is **an OSS library that the 3D vision community adopts as the
  default eval harness.** Credibility comes from reproducing published
  numbers, not from inventing new metrics.
- Budget: minimal. Develop on CPU where possible; only rent a GPU when
  actually running inference. Target v0.1 under $100 of cloud compute.
- Audience: grad students and researchers. CLI-first, Python-first. No web UI
  in v0.1.

## 1. Non-goals (do not scope-creep into these)

- Training or fine-tuning models. Inference only.
- A novel metric. Only implement what papers already report.
- A web UI, leaderboard site, or hosted service. v0.2+.
- Distributed / multi-node. Single GPU, single node.
- A new dataset. Use existing ones.
- Auto-downloading gated datasets. Provide scripts + instructions; user
  handles auth.

## 2. v0.1 scope

Scope is defined by what's empirically demonstrated, not a pre-committed
shortlist. See [REPRODUCTIONS.md](./REPRODUCTIONS.md) for the live status
matrix; this section describes scope, that file describes state.

**Demonstrated v0.1 surface (post 2026-04-19 pivot):**

13 ✅ paper-match cells across NYU + KITTI mono depth (DA-V2 S/B/L,
Metric3Dv2 L/Giant, MoGe-1 ViT-L, Marigold v1-1, DA3) plus a multi-view
chamfer track (VGGT/DA3 on ETH3D + DTU) and a pose-sweep track
(VGGT/DA3 on ETH3D). Adapters shipped: DA-V2, Metric3Dv2, MASt3R, VGGT,
DA3, MoGe-1, MoGe-2, Marigold, GeoWizard, Depth Pro, π³.

**Datasets:**

- **NYUv2** — primary mono-depth bench (8 ✅ cells)
- **KITTI** — outdoor mono-depth, Eigen + Garg (5 ✅ cells)
- **ETH3D high-res** — multi-view chamfer + pose
- **DTU MVS** — VGGT chamfer track
- **DIODE, GSO, Co3Dv2, 7Scenes, iBims-1** — secondary (loaders shipped,
  paper-row work ongoing)

**Deprioritized 2026-04-19 (auth-gated, infra ready):** Sintel,
ScanNet / ScanNet-1500. Loaders work; substitutes promoted (GSO/iBims-1
for synthetic clean-GT, Co3Dv2/7Scenes for pose).

**Tasks:** monocular depth, multi-view depth/chamfer, relative camera
pose.

**Acceptance criterion for v0.1:**

```
$ plumbline reproduce vggt-paper-dtu-mvs
```

...runs VGGT on DTU MVS and produces chamfer (overall) within ±5 % of
published 0.382 m (VGGT paper Table 2). Currently blocked by D3
(per-view-masked vs scene-merged metric shape — needs CUT3R reference-
code diff per § 12 to close). De facto gate while D3 is open: the 13 ✅
mono-depth cells already landed. v0.1 ships when D3 (and the parallel
D4 ETH3D track) match paper, or when both are explicitly demoted to
"protocol mismatch — informational" with that reasoning recorded in the
YAML.

**Until D3/D4 close, no chamfer reproduction is launched on a full
dataset.** Single-record diff (§ 12) is the only path forward.

## 3. Canonical conventions (non-negotiable)

Every model adapter and dataset loader must convert to these. This is the
whole value prop of the harness — do not let leakage happen.

- **Camera convention:** OpenCV. Right-handed, +X right, +Y down, +Z forward
  (into the scene). Image origin top-left.
- **World frame:** First camera of the sequence is the world frame. `R = I`,
  `t = 0` for camera 0.
- **Extrinsics:** `world_from_camera`, shape `(4, 4)`. Document this
  explicitly; it's the opposite of what some papers use.
- **Intrinsics:** `K` shape `(3, 3)`, pixels, fx/fy/cx/cy with standard
  layout. No normalized coords.
- **Depth:** `(H, W)` float32, meters when metric, dimensionless otherwise.
  Zero or NaN = invalid.
- **Point map:** `(H, W, 3)` float32 in the world frame.
- **Image:** `(H, W, 3)` uint8, sRGB, no alpha. Linear color is a v0.2
  concern (but flag it in the schema so it can be added without breaking).
- **Resolution:** store GT at native resolution; resize predictions to GT for
  metric computation, not the other way around.

Put these in `plumbline/conventions.py` with docstrings and assertion
helpers (`assert_valid_extrinsics`, `assert_valid_depth`, etc.). Use them
everywhere.

## 4. Architecture (4 layers)

```
plumbline/
├── conventions.py          # canonical schemas + assertions
├── models/
│   ├── base.py             # Model ABC + Prediction dataclass
│   ├── registry.py         # Model.from_hub("name") dispatch
│   ├── depth_anything_v2.py
│   ├── metric3d_v2.py
│   ├── mast3r.py
│   ├── vggt.py
│   └── depth_anything_3.py
├── datasets/
│   ├── base.py             # Dataset ABC + Sample dataclass
│   ├── registry.py
│   ├── sintel.py
│   ├── scannet.py
│   └── eth3d.py
├── metrics/
│   ├── depth.py            # AbsRel, RMSE, δ₁/₂/₃, SILog
│   ├── pose.py             # R/t error, AUC@5/10/30, ATE
│   ├── pointmap.py         # Chamfer, F-score
│   └── alignment.py        # scale alignment modes
├── runner.py               # main evaluate() loop, caching, OOM recovery
├── report.py               # markdown / json / html output
├── cache.py                # prediction cache (key = model+dataset+config hash)
├── cli.py                  # `plumbline` command
└── reproductions/
    └── vggt_scannet_depth.yaml   # paper-number configs
```

### 4.1 Model adapter interface

```python
# plumbline/models/base.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Prediction:
    depth: Optional[np.ndarray] = None            # (N, H, W) float32
    intrinsics: Optional[np.ndarray] = None       # (N, 3, 3) float32
    extrinsics: Optional[np.ndarray] = None       # (N, 4, 4) world_from_camera
    point_map: Optional[np.ndarray] = None        # (N, H, W, 3) float32, world frame
    confidence: Optional[np.ndarray] = None       # (N, H, W) float32, [0,1]
    metadata: dict = None                         # runtime_ms, peak_vram_mb, etc.

@dataclass
class ModelCapabilities:
    tasks: set[str]                 # {"mono_depth", "mvs_depth", "pose", ...}
    is_metric: bool
    min_views: int
    max_views: int                  # use math.inf if unbounded
    requires_intrinsics: bool
    default_resolution: tuple[int, int]  # (H, W) the model was trained at

class Model(ABC):
    name: str
    capabilities: ModelCapabilities

    @abstractmethod
    def predict(
        self,
        images: np.ndarray,              # (N, H, W, 3) uint8 sRGB
        intrinsics: Optional[np.ndarray] = None,  # (N, 3, 3) or None
    ) -> Prediction: ...

    @classmethod
    def from_hub(cls, name: str, device: str = "cuda") -> "Model":
        return MODEL_REGISTRY[name](device=device)
```

**Adapter implementation rules:**
- The adapter owns the device, dtype, resize, normalization, and all
  model-specific preprocessing. The caller never deals with torch.
- The adapter is responsible for converting the model's native output to
  canonical conventions. Document every flip/transpose/scale with a comment
  citing the source (paper section, model repo file + line).
- The adapter declares what it supports. If a model can't do pose, it
  returns `extrinsics=None`; the runner skips pose metrics for that model.
- Weight files live in `~/.cache/plumbline/weights/<model>/`. Use the model's
  official HF hub location; do not re-upload weights.

### 4.2 Dataset loader interface

```python
# plumbline/datasets/base.py
@dataclass
class Sample:
    sample_id: str                  # stable, deterministic, used as cache key
    images: np.ndarray              # (N, H, W, 3) uint8 sRGB
    intrinsics: np.ndarray          # (N, 3, 3) float32
    extrinsics_gt: np.ndarray       # (N, 4, 4) world_from_camera
    depth_gt: Optional[np.ndarray]  # (N, H, W) float32 meters
    depth_valid: Optional[np.ndarray]  # (N, H, W) bool
    point_cloud_gt: Optional[np.ndarray]  # (M, 3) world frame
    metadata: dict                  # scene_id, split, difficulty, etc.

class Dataset(ABC):
    name: str
    split: str

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...
    @abstractmethod
    def __len__(self) -> int: ...
```

**Loader implementation rules:**
- Do the coordinate conversion once, at load time. Never inside the runner.
- Pre-compute and cache a manifest (JSON) listing sample IDs and file paths.
  Iteration reads from the manifest, not from a directory scan.
- Provide a `subset(n)` method for quick dev runs. Use deterministic
  sampling (sort + stride, not random).
- If the dataset requires auth/manual download, the loader raises a clear
  error with the URL and expected path layout on first use.

### 4.3 Metrics

Pure functions. Inputs are canonical tensors. No torch, numpy only.

```python
# plumbline/metrics/depth.py
def abs_rel(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float: ...
def delta_threshold(pred, gt, valid, threshold=1.25) -> float: ...
def rmse(pred, gt, valid) -> float: ...
def silog(pred, gt, valid) -> float: ...

# plumbline/metrics/alignment.py
def align_scale_median(pred, gt, valid) -> np.ndarray: ...
def align_scale_lstsq(pred, gt, valid) -> np.ndarray: ...
def align_scale_and_shift(pred, gt, valid) -> tuple[np.ndarray, float, float]: ...
```

Scale alignment is a first-class concept. The runner must log which mode was
used; the report displays it; cached predictions store raw (unaligned)
values so the alignment can be changed without re-running inference.

### 4.4 Runner

```python
def evaluate(
    model: Model,
    dataset: Dataset,
    tasks: list[str],
    scale_alignment: str = "median",
    max_views: int = 8,
    device: str = "cuda:0",
    cache_dir: Path = DEFAULT_CACHE,
) -> Report:
    ...
```

Responsibilities:
1. Iterate samples from the dataset.
2. For each sample, compute a cache key from
   `(model.name, model.version, dataset.name, sample_id, config_hash)`. If
   predictions exist on disk, load them; otherwise run inference and save.
3. Compute metrics from (possibly cached) predictions + GT.
4. OOM recovery: catch `torch.cuda.OutOfMemoryError`, log the sample, skip
   it, continue. Report logs N_skipped.
5. Determinism: seed everything, log GPU model, CUDA version, torch version,
   model checkpoint hash.

### 4.5 Report

Three output formats from the same `Report` object:
- `to_markdown()` — for the terminal and README embedding.
- `to_json()` — machine-readable, stable schema. This is what a future
  leaderboard consumes.
- `to_html()` — v0.2. Skip for v0.1.

The JSON schema is public API; version it (`schema_version: "1.0.0"`) so
future changes don't break consumers.

## 5. Caching strategy (critical)

Inference is the expensive thing. Everything else is free. Structure the
code so that:

- Raw predictions (depth, pose, point map) are always cached to disk after
  inference.
- Cache key includes model version + checkpoint hash + preprocessing config.
- Changing a metric, an alignment mode, or a report format **never**
  triggers reinference.
- `plumbline clear-cache` exists and is selective (`--model vggt`,
  `--dataset scannet`).
- Store predictions as compressed npz (`np.savez_compressed`) in
  `~/.cache/plumbline/predictions/<model>/<dataset>/<sample_id>.npz`.

This one piece of infrastructure is what makes the harness usable on a
laptop-plus-occasional-GPU budget.

## 6. CLI surface (v0.1)

```
plumbline list-models
plumbline list-datasets
plumbline run --model vggt --dataset dtu --tasks mvs_depth
plumbline reproduce vggt-paper-dtu-mvs
plumbline report --json results.json
plumbline clear-cache [--model X] [--dataset Y]
```

Use `click` or `typer`. Not `argparse`.

## 7. Build history (historical — see git tags)

The original 5-week build plan executed roughly as designed; the
skeleton, conventions, and first three models landed by 2026-03 and the
ETH3D + KITTI work landed during the 2026-04-19 GPU-validation session.
The week-by-week plan is preserved in git history pre-2026-04-25 if
needed. Current state is in `REPRODUCTIONS.md` (what's matched) and
`docs/DISCREPANCIES.md` (what's open). Total v0.1 compute spend to date
is well under the original ~$100 envelope.

## 8. What to hand off to a collaborator vs do yourself

Handoff-friendly (have someone else do these):
- New dataset loaders (after base class is solid).
- New model adapters (after two existing ones set the pattern).
- Metric implementations (well-specified pure functions).
- Documentation.

Don't hand off:
- The conventions module and its enforcement.
- The caching layer.
- The reproduction configs and their tolerance numbers.
- The JSON report schema.

These are the parts that determine whether the harness is trusted.

## 9. Known traps (read before starting)

- **Extrinsic conventions.** Half the 3D world uses `camera_from_world`, the
  other half uses `world_from_camera`. Every dataset and every model is a
  potential inversion bug. Write the assertion helpers first, use them
  everywhere.
- **Depth vs inverse depth.** Some models predict 1/z, some predict z, some
  predict disparity. Document which one the adapter receives natively and
  convert explicitly with a comment.
- **Image resize interpolation.** Bilinear for images, nearest for depth GT,
  nearest for masks. Never bilinear-resize a depth map.
- **Focal length and aspect ratio.** When a model resizes internally, the
  effective focal length changes. Adapters must unscale predicted
  intrinsics back to input-image pixels before returning.
- **ScanNet's extrinsics files.** There's a well-known issue where some
  poses are `inf`; filter them in the loader.
- **Sintel's intrinsics.** They're provided but some tools assume a specific
  focal length; verify against the Sintel README.
- **Non-determinism in VGGT.** Some ops are non-deterministic on CUDA
  even with seeds; document the tolerance and don't chase bitwise
  reproducibility.

## 10. v0.2 roadmap

Open work, in rough priority. Each line is a pointer; `docs/DISCREPANCIES.md`
has the live diagnosis state.

**Tier 1 — close v0.1 gate:**
- **D3** VGGT-DTU chamfer (per-view-masked vs scene-merged) — § 12 diff against CUT3R `eval/mv_recon/`.
- **D4** VGGT-ETH3D multiscene — same root cause as D3, same fix.
- **D10** ETH3D 13-scene full split (or demote to informational with larger tolerance).

**Tier 2 — paper-row unlocks (each is a single-record-diff sprint):**
- DIODE outdoor protocol (clip + sky-mask).
- MoGe-2 metric eval — extend to DIODE + KITTI.
- Pose benchmarks — Co3Dv2, 7Scenes (loaders shipped, paper rows pending).
- Depth Pro paper rows (Sun-RGBD or projected-ETH3D — paper doesn't eval NYU).
- MASt3R, π³, GeoWizard inference smoke-tests.
- New adapters: CUT3R, Fast3R, FLARE, MapAnything, MonST3R, DepthFM.

**Tier 3 — systems / structural:**
- Paper-protocol presets (`protocol: nyu_eigen_2014` expands to the exact tuple).
- Failure-mode diagnostic flags (`plumbline reproduce --diagnose`).
- Cache-key GT-side fingerprinting (D21 covered the input side).

**Out of scope for v0.2:** training, novel metrics, web UI / leaderboard,
distributed eval, novel-view synthesis, point tracking, uncertainty
calibration.

## 11. First thing to do when you start

The harness exists. The starting question is no longer "how do I build
it" but "which open paper-row do I close next, and how". Answer:

1. Read `docs/DISCREPANCIES.md` § Open issues to pick a target.
2. Follow § 12 (single-record diff protocol) to close it.
3. Update `REPRODUCTIONS.md` with the new state.

If you are bringing up a fresh GPU box, see `GPU_RUNBOOK.md` — but
follow § 12's thrift rules: don't bulk-pull the dataset cache, pull
only the records under investigation.

## 12. Reproduction protocol — single-record diff

Don't run a model on a whole dataset to discover a 130× discrepancy.
Pick one sample, clone the reference repo, run both pipelines on that
one sample, and diff the intermediate tensors stage by stage. The first
diverging stage is the bug.

**Stages to diff (in order):**

1. **Sample loading** — image bytes, GT bytes. Hash the raw files.
2. **Image preprocessing** — resize policy, crop, normalization,
   dtype. Compare tensor shape + value range + a 1 KB byte sample.
3. **Model input** — exactly what hits `model.forward()`. Shape,
   dtype, device, value range.
4. **Model output (raw)** — depth / disparity / point map straight
   from the model, before any postprocess. This is where weight or
   architecture mismatches show up.
5. **Postprocess** — alignment mode, scale, shift, clamp, mask,
   units. The chamfer fights live here.
6. **GT preprocessing** — same crops/masks as the prediction.
   Off-by-one cropping is a common silent bug.
7. **Metric computation** — per-pixel error → aggregate. Log the
   pre-aggregate tensor, not just the scalar.

At each stage save the tensor as `.npy` (and a hash of it), compare
plumbline vs reference. When they diverge, fix the upstream stage
*before* moving on — divergence compounds.

**Reference repos for the open issues:**

- D3 / D4 (VGGT chamfer) — CUT3R `eval/mv_recon/`
- D9 / D18 / D22 (Marigold/GeoWizard KITTI) — `prs-eth/Marigold/src/dataset/kitti_dataset.py`
- D8 (MoGe-KITTI — closed) — already done this way

**Thrift rules for GPU bring-up:**

- Bootstrap pulls *only* the sample being diffed, not the dataset.
  S3 layout supports per-sample selective pull
  (`s3://plumbline-bench/datasets/<name>/<sample_id>/*`).
- Pull *only* the one model's weights for the issue under
  investigation.
- Don't warm the prediction cache for unrelated reproductions.
- The full-dataset run is the *last* step, after a single-record diff
  has shown stages 1–7 match within numerical tolerance.

If the reference repo doesn't exist or doesn't run, that's an
upstream-blocked issue (e.g. D22) — document and demote, don't
guess.
