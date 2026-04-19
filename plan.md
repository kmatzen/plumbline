# plumbline — Implementation Plan

A reproducible evaluation harness for 3D geometric foundation models. Think
`lm-evaluation-harness` but for models like VGGT, Depth Anything 3, MASt3R,
Metric3Dv2, Depth Pro, MoGe.

This document is the spec. Work through it section by section. Ask before
deviating from the architecture; feel free to deviate on implementation
details within each section.

> **Status banner (2026-04-19)** — the original v0.1 gate
> (`reproduce vggt-paper-scannet-depth`) has been **retargeted** to
> `reproduce vggt-paper-dtu-mvs`. The VGGT paper doesn't actually
> report ScanNet depth (Table 2 is DTU; ScanNet-1500 is two-view
> matching, not depth). The DTU gate pins Table 2's chamfer=0.382 and
> is unblocked on data (DTU is public, no ToS) but blocked on the DTU
> loader, which arrives in a follow-up. In the meantime the harness
> ships **7 paper-match reproductions on NYUv2** across four model
> families (DA-V2 S/B/L, DA-V2 Metric-Indoor-L, Metric3Dv2 L/Giant2,
> DA3), plus informational ETH3D courtyard pose sweeps for VGGT and
> DA3 and chamfer reproduction configs for both. See
> [REPRODUCTIONS.md](./REPRODUCTIONS.md) for the live status matrix.

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

Five models, three datasets, three tasks, one command that reproduces a
published number.

**Models (v0.1):**
1. Depth Anything V2 (monocular, relative depth) — easiest, fastest, use as
   the shakedown model.
2. Metric3Dv2 (monocular, metric depth + normals).
3. MASt3R (multi-view, pair-based).
4. VGGT (multi-view, feed-forward, up to ~32 views).
5. Depth Anything 3 (multi-view, newest).

**Datasets (v0.1):**
1. Sintel — synthetic, tiny, perfect GT. Use for correctness shakedown.
2. ScanNet v2 (test split) — indoor, widely cited. Depth + pose.
3. ETH3D (high-res multi-view) — outdoor, hard. Multi-view stereo.

**Tasks (v0.1):**
1. Monocular depth estimation.
2. Multi-view depth estimation.
3. Relative camera pose estimation.

**Acceptance criterion for v0.1:**

```
$ plumbline reproduce vggt-paper-dtu-mvs
```

...runs VGGT on the DTU MVS test set and produces chamfer (overall)
within ±5% of the published 0.382 (VGGT paper Table 2, no-GT-camera
block). Originally this gate targeted ScanNet depth, but the VGGT
paper does not evaluate on ScanNet depth — Table 2 (DTU) is the real
depth/point-map table; ScanNet-1500 (Table 4) is two-view matching.
See § 11 below for the history; see
[REPRODUCTIONS.md](./REPRODUCTIONS.md) for the live status. When the
DTU gate lands within tolerance, v0.1 ships.

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

## 7. Week-by-week plan

Each week ends with a commit tagged `v0.1-week-N` and a working demo. Don't
skip ahead; each layer depends on the previous one being solid.

### Week 1 — skeleton, conventions, and Sintel + Depth Anything V2

Goal: end-to-end pipeline on the easiest model + easiest dataset. No GPU
needed for most of this week; rent one at the end for the actual inference
run.

- [ ] Set up repo: `pyproject.toml` (uv or poetry), `ruff`, `pytest`, CI on
      GitHub Actions running lint + tests on CPU.
- [ ] Write `conventions.py` with full assertion helpers. Unit test them.
- [ ] Implement `Prediction`, `Sample`, `Model`, `Dataset` base classes.
- [ ] Implement Sintel loader. Unit test with synthetic tensors first, then
      real data. Confirm extrinsics are world-from-camera in OpenCV.
- [ ] Implement Depth Anything V2 adapter. It's monocular + relative, so
      scope is small.
- [ ] Implement depth metrics (`abs_rel`, `rmse`, `delta_threshold`) and
      median scale alignment.
- [ ] Implement minimal `runner.evaluate()` with caching.
- [ ] Implement `report.to_markdown()` and `report.to_json()`.
- [ ] `plumbline run --model depth-anything-v2 --dataset sintel` produces a
      number. Commit. Tag.

**GPU spend this week:** ~2 hours on a 4090 (~$1). Everything else is CPU.

### Week 2 — ScanNet + Metric3Dv2, scale alignment modes

- [ ] Implement ScanNet v2 test-split loader. Handle the intrinsics and
      pose file formats. Validate by reprojecting a GT point cloud through
      GT poses and checking alignment with a GT depth map.
- [ ] Implement Metric3Dv2 adapter. This model is metric, so you'll exercise
      the "no scale alignment" path for the first time.
- [ ] Add `align_scale_lstsq` and `align_scale_and_shift` modes.
- [ ] Add SILog metric.
- [ ] Expand `report.to_markdown()` to show alignment mode in the output.
- [ ] Write integration test: DepthAnythingV2 on Sintel subset gives a
      deterministic number across two runs.
- [ ] Commit. Tag.

**GPU spend:** ~8 hours on a 4090 (~$4).

### Week 3 — multi-view: MASt3R + pose metrics

This is where the architecture gets stress-tested. Multi-view models have
different input shapes, different output formats, and pose.

- [ ] Extend `Prediction` and `Sample` handling for N-view input.
- [ ] Implement MASt3R adapter. Pairwise → pair reasoning in the adapter, or
      handle pairs at the runner level? Recommendation: adapter handles its
      own pair batching internally; runner just hands it N views.
- [ ] Implement pose metrics: rotation error (geodesic), translation error
      (cosine + magnitude when metric), AUC@5°/10°/30°.
- [ ] Add pose evaluation on ScanNet.
- [ ] Handle the world-frame convention: GT poses must be re-referenced to
      the first camera of the sampled view-set, not the dataset's global
      frame. Unit test this explicitly — it's a classic source of bugs.
- [ ] Commit. Tag.

**GPU spend:** ~15 hours on L40S or A100 (~$10).

### Week 4 — VGGT, ETH3D, reproduction config

- [ ] Implement VGGT adapter. It's the biggest model; validate VRAM usage
      fits in 24GB at the paper's default view count.
- [ ] Implement ETH3D multi-view loader.
- [ ] Pick one specific number from the VGGT paper. **Status: Table 2 (DTU
      dense MVS, chamfer=0.382) is the real depth/point-map target**; the
      original plan referenced ScanNet depth which the paper does not
      report. `reproductions/vggt_dtu_mvs.yaml` pins the DTU target.
- [ ] Run `plumbline reproduce vggt-paper-dtu-mvs`. Debug until the
      chamfer is within ±5% of 0.382. **This is the v0.1 gate.**
- [ ] Write `REPRODUCTIONS.md` documenting the exact procedure and the
      published reference.
- [ ] Commit. Tag `v0.1.0`.

**GPU spend:** ~25 hours on a mix of 4090 and A100 (~$20).

### Week 5 — Depth Anything 3, polish, README, first release

- [ ] Implement Depth Anything 3 adapter.
- [ ] Write a real README. Include one-line install, 30-second quickstart,
      the reproduction command, the supported model/dataset matrix.
- [ ] Add `pytest` suite that runs on CI against tiny synthetic data for
      every model adapter (smoke tests only — no GPU on CI).
- [ ] Add a `CONTRIBUTING.md` explaining how to write a new model adapter.
      This document is what turns the project from "Kevin's repo" into a
      community project.
- [ ] Publish to PyPI as `plumbline-bench` (the `plumbline` import name
      remains; `plumbline` on PyPI was taken).
- [ ] Cut a v0.1.0 GitHub release. Write a short blog post or arXiv note
      announcing it.

**GPU spend:** ~10 hours (~$5).

### Total v0.1 budget

- Compute: ~$40.
- Storage: ~$15 (datasets on a persistent volume during the 5 weeks).
- Slack + mistakes: ~$40.
- **Total: ~$100.**

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

## 10. Out-of-scope parking lot (v0.2 and beyond)

Capture ideas here instead of building them:
- Novel-view synthesis evaluation (PSNR/SSIM/LPIPS).
- Point tracking evaluation.
- Uncertainty calibration metrics.
- Failure-case browser web UI.
- Nightly CI running the full suite.
- Hosted leaderboard site.
- Additional models: π³, MoGe, CUT3R, MonST3R, Fast3R, Depth Pro, MapAnything
  backends.
- Additional datasets: TUM-dynamics, CO3Dv2, KITTI, DTU, NYUv2, Tanks &
  Temples, Replica.
- HDR / linear-color evaluation path (leverages framewright).
- Distributed eval across multiple GPUs.

## 11. First thing to do when you start

Before writing any code:
1. Read the VGGT, Depth Anything 3, and MASt3R papers for 20 minutes each.
   Focus on their evaluation sections. Note exactly which scale alignment
   they use, which ScanNet split, which view count.
2. Clone their official repos. Run their provided demo scripts on a single
   image. This confirms the environment and gives a reference for the
   adapter.
3. Create the repo skeleton from section 4 with empty files and stub
   classes. Commit. This makes the plan concrete before any real
   implementation decisions.

Then start Week 1.
