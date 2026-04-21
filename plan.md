# plumbline — Implementation Plan

A reproducible evaluation harness for 3D geometric foundation models. Think
`lm-evaluation-harness` but for models like VGGT, Depth Anything 3, MASt3R,
Metric3Dv2, Depth Pro, MoGe.

This document is the spec. Work through it section by section. Ask before
deviating from the architecture; feel free to deviate on implementation
details within each section.

> **Status banner (2026-04-21, after multi-day laptop-prep session)** —
>
> The GPU rental session is fully staged. Every laptop-doable piece
> of the "flesh out the catalog + pre-fetch everything" work has
> landed; the remaining gate is booking the rental box.
>
> **What's ready to run on the rental box:**
>
> - **21 verified-PDF paper-match reproductions** in the validation
>   queue (`scripts/list_validation_targets.py`), covering DA-V2 S/B/L
>   on NYU+KITTI, Metric3D-v2 L+Giant on NYU+KITTI, MoGe-1 on NYU+
>   KITTI+DIODE (indoor+both), Marigold v1-1 on NYU+KITTI, DA3 on NYU,
>   GeoWizard on NYU+KITTI, VGGT Table 2 DTU (the v0.1 gate), VGGT
>   Table 3 ETH3D 3-scene.
> - **Two informational multi-view smoke YAMLs** for π³ on DTU +
>   ETH3D (unpinned — first-run observation becomes the basis for
>   later paper-cell pinning once Pi3's unit convention is confirmed).
> - **S3 cache** at `s3://plumbline-bench/` (us-west-2) with
>   7,287 objects / 54 GB total: datasets (12 GB: NYUv2, KITTI
>   Eigen-652 pruned, DTU 22-scan test + GT Points, ETH3D 3-scene,
>   iBims-1, GSO), hf-cache (35 GB: 7 model repos with `-hf` suffixes
>   where the adapter expects them), torch-hub-cache (7.4 GB: all
>   3 Metric3D-v2 ViT variants + the YvanYin repo tree). Agent
>   pulls everything with `scripts/stage_all_data.sh`.
> - **Protocol presets** (`protocols/*.yaml`) encode every published
>   protocol this harness touches (NYU Eigen-2014, KITTI Eigen-Garg,
>   DIODE MoGe, ETH3D VGGT Table 3, DTU VGGT Table 2, GSO MoGe).
>   Reproductions declare `protocol: <name>` and inherit the fixed
>   fields; the runner raises `ProtocolConflictError` on drift.
> - **Trust hardening from the 2026-04-20 audit
>   (`reproductions/AUDIT.md`)**: every pinned `paper_reference.value`
>   tagged `source_confidence: verified_pdf` only if the cell was
>   confirmed against the arXiv PDF. Four fabricated/wrong-row claims
>   were downgraded to `value: null` (depth_pro_nyuv2,
>   depth_anything_v2_sintel, moge2_vitl_kitti, da_v2_small_diode_indoor);
>   five wrong-table citations were corrected; one wrong-value
>   (moge_vitl_kitti 0.0405 → 0.0408) was corrected. The
>   `REPRODUCTIONS.md` status matrix now counts only verified_pdf
>   cells as ✅.
> - **Autonomous-agent runbook** at `docs/AGENT_GPU_RUNBOOK.md` with
>   a hard-constraints list (never modify YAMLs, never commit, never
>   invent numbers, never delete the S3 cache). The agent's
>   validation loop pushes JSON per reproduction + a markdown report
>   to `s3://plumbline-bench/runs/<ts>/`.
>
> **Known gates before the GPU session:**
>
> 1. Three adapters require a clone (no PyPI): GeoWizard at
>    `$GEOWIZARD_ROOT`, π³ at `$PI3_ROOT`, MASt3R at `$MAST3R_ROOT`.
>    The agent runbook documents all three in § "Per-adapter
>    first-run notes" via the human `GPU_RUNBOOK.md`.
> 2. Metric3D-v2 needs `mmengine` + `mmcv-lite` in the venv (see
>    `GPU_RUNBOOK.md § Metric3Dv2`) — torch.hub weights are cached
>    but the code deps aren't on PyPI's `models` extra.
> 3. HF auth is optional but recommended (`HF_TOKEN` env var +
>    `hf auth login`) — anonymous downloads work but rate-limit.
>    The agent's pre-flight picks this up.
>
> **Out-of-scope for this session (v0.2+ items deferred):**
>
> - Paper-row pinning for Pi3 multi-view chamfer (Pi3's Table 3
>   DTU/ETH3D metrics differ ~4× from VGGT's — units/normalization
>   need first-run observation).
> - Depth Pro paper rows: paper evaluates Booster/ETH3D/Middlebury/
>   NuScenes/Sintel/Sun-RGBD only — NYU/KITTI Depth Pro YAMLs are
>   informational. Would require a new dataset loader.
> - 7-Scenes / Co3Dv2 pose benchmarks: loaders wired, data not
>   staged (7-Scenes 12 GB, Co3Dv2 needs a pinned sequence subset
>   first).
> - MoGe-2 variants, MASt3R, Pi3 HF weights: none cached to S3
>   (all informational-only reproductions currently).
>
> See [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) for the per-YAML
> status matrix and [`docs/AGENT_GPU_RUNBOOK.md`](./docs/AGENT_GPU_RUNBOOK.md)
> for the autonomous-agent playbook. § 10 below has the revised
> v0.2 roadmap.
>
> ----
>
> **Historical note — earlier 2026-04-19 session (first real GPU
> validation)**: spent an RTX 3090 Ti session pinning real numbers
> across 6 model adapters × 5 datasets. Landed 7 NYU paper-matches
> (DA-V2 S/B/L + Metric-Indoor-L, Metric3Dv2 L/Giant2, DA3),
> 2 DIODE indoor effective-matches (DA-V2-small, MoGe-1 ROE), the
> MoGe NYU row under ROE alignment (0.0305 vs paper 0.0297), and
> first KITTI paper-match (DA-V2-small 0.077 vs 0.078). Major session
> wins: `scale_shift_robust` (ROE) alignment closed MoGe NYU to paper;
> ICP alignment mode + chamfer outlier-mask + depth→point-map
> back-projection all shipped as infrastructure. Major session find:
> an earlier celebration of a "500× ETH3D F-score improvement" was a
> unit misread (f_score returns percent, not fraction); the real
> improvement was 5.6×. See `git log origin/main -- REPRODUCTIONS.md`
> for the exact commits from that run.

---

## 0. Context for the agent

- Owner: Kevin Blackburn-Matzen (formerly Facebook CompPhoto/XR). Deep
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

## 10. v0.2 roadmap (ordered by paper-gap leverage)

Revised 2026-04-19 after a GPU-validation session pinned 16 reproductions
on an RTX 3090 Ti. The session surfaced a clearer picture of what's
missing and which items matter most for reproducing published tables.

### Tier 1 — close known paper gaps

1. **ETH3D protocol rewrite — LANDED 2026-04-20.** `aggregation:
   scene` + Acc/Comp/Overall (distance metres, 1 cm voxel) now
   reproduces VGGT Table 3 within 15% on a 3-scene subset (Overall
   0.818 vs paper 0.709; see `vggt-eth3d-multiscene-chamfer`).
   The earlier "papers report 60-90% F-score" framing was a
   unit-interpretation error — VGGT Table 3 is distance metrics,
   not F-score, and F@5cm is an indoor T&T threshold unsuited to
   outdoor ETH3D. Remaining work: port the same scene-merge path
   to DTU (mm units, different GT format); extend to more ETH3D
   scenes when disk permits.

2. **KITTI reproduction.** Loader exists; data partially downloaded
   (annotated-depth GT on disk at `/home/claude/data/kitti`). Needs:
   per-drive raw imagery for the ~28 drives in the Eigen-with-GT test
   split, plus a pinned sample list. Unblocks DA-V2 / Metric3Dv2 /
   MoGe KITTI rows — at least 3 paper rows per model.

3. **DIODE outdoor protocol.** Indoor-only matches paper reasonably
   (AbsRel 0.0465 observed vs 0.0313 paper under ROE); combined
   (indoor + outdoor) is 6.4× off (0.1993 vs 0.0313). Outdoor is the
   hard slice — suspected wrong `depth_clip` (we use 50 m; DIODE
   outdoor GT ranges to 300 m) and/or missing sky-mask. Read MoGe's
   DIODE eval protocol in their repo (`moge/test/metrics.py` has it)
   and match.

### Tier 2 — new paper-row unlocks

4. **Diffusion / generative depth models.** Marigold landed in this
   session (adapter wired, NYU reproduction running). Follow-ups:
   - **GeoWizard** (Fu et al. 2024): diffusion depth + normals.
     **Adapter shipped 2026-04-20** (lazy-import via `$GEOWIZARD_ROOT`).
     Inference smoke-test pending GPU rental.
   - **DepthFM** (Gui et al. 2024): flow-matching-based depth.
   - **Depth Pro** (Apple, Bochkovskii et al. 2024): not diffusion
     but a recent HF-available metric model; fits the mono-depth
     shape. **Adapter wired 2026-04-19; original A9 plan to expand
     paper rows on KITTI/NYU was killed by the 2026-04-20 audit** —
     Depth Pro paper Table 1 evaluates only Booster, ETH3D, Middlebury,
     NuScenes, Sintel, Sun-RGBD (δ₁ values 40-89). To unlock real
     Depth Pro paper rows, choose:
       - **(a) Sun-RGBD loader** (~5 GB, semi-public; new dataset module).
         Targets δ₁=89.0 — easiest cell to reproduce. Estimated 1 day.
       - **(b) Extend ETH3DDataset to project the laser-scan point
         cloud through each camera's pose to a per-image GT depth
         map.** Targets δ₁=41.5. Reuses existing ETH3D infrastructure
         but the projection + visibility step is non-trivial (occlusion
         culling, masking). Estimated 1-2 days.
       - **(c) Sintel-depth** (already have RGB loader; depth + cam
         archives are auth-gated). Targets δ₁=40.0. Blocked on
         user's TUM email.
     None are on the v0.1 critical path; pick during a focused Depth
     Pro paper-row sprint, not a side-effort.
   - **GeoCrafter** (if different from GeoWizard — confirm) and
     video-consistent diffusion depth models.

5. **Multi-view 3D foundation models** (the recent wave beyond
   VGGT/MASt3R/DA3):
   - **π³ (Pi-Cubed / PI3)**: multi-view from Bytedance (2024).
     **Adapter shipped 2026-04-20** (lazy-import via `$PI3_ROOT`).
     Two variants wired (`pi3` original + `pi3x` Dec-2025 rev).
     Inference smoke-test pending GPU rental.
   - **CUT3R**: "Continuous 3D Tokenizer" (Wang et al. 2025).
     Stateful, ingests a sequence of views.
   - **Fast3R**: large-ensemble 3D from Meta (2024/2025). Single
     forward pass on many views.
   - **FLARE**: [confirm — possibly FLARE by Meng et al. 2025 for
     fast multi-view reconstruction; check canonical repo].
   - **MapAnything**: unified 3D reconstruction w/ map prior (Wang
     et al. 2025).
   - **MonST3R**: dynamic-scene variant of DUSt3R. Multi-frame video
     input.
   All of these fit the ``mvs_depth`` / ``pose`` capability shape
   but individually have upstream-install quirks — each is ~1 day of
   adapter work + smoke-test.

6. **Pose benchmarks — pivoted to Co3Dv2 + 7Scenes.** Infrastructure
   (rotation / translation / AUC@5/10/30° metrics, pairwise relative
   pose) exists; the loaders that map to paper pose tables don't.
   ScanNet-1500 is deprioritized (see 2026-04-19 pivot below); the
   active Tier-2 pose-benchmark targets are:
   - **Co3Dv2** (Meta, public, no auth) — VGGT Table 1, DUSt3R,
     MASt3R all report on it. Object-centric, ~1.5M frames across
     ~19K sequences. The flagship pose benchmark that unblocks
     real paper-row reproductions.
   - **7Scenes** (Microsoft Research, public) — classical
     relocalization, still cited in MASt3R era.
   - **TUM-RGBD** (SLAM-style trajectory, public).

   Deprioritized:
   - **ScanNet-1500** — loader is wired and unit-tested (shipped
     in 2026-04-19 session) but the image data is ScanNet-ToS-gated
     and the auth email hasn't landed. Remains usable the moment
     access arrives; just not on the critical path for "good
     benchmark" because Co3Dv2 covers the same protocol space.
   - **RealEstate10K** — trajectory, paid-to-download RGB.

7. **MoGe-2 metric eval**. Already have the config
   (`moge2-vitl-nyuv2-metric`, observed AbsRel=0.0899 without
   alignment). Paper reports MoGe-2 improvements under
   `scale_alignment: none` protocol on benchmarks with GT intrinsics
   — extend to DIODE + KITTI + more indoor benchmarks.

### Tier 3 — systems / structural

7. **Paper-protocol presets.** Each published table uses a specific
   combo of (crop, alignment, depth_clip, valid-mask). Today every
   plumbline YAML picks these ad-hoc and documents deviations in
   notes. Cleaner: a `protocol: nyu_eigen_2014` block that expands
   to the exact tuple; reproduction YAMLs declare which protocol
   they follow and can't accidentally mis-align.

8. **Failure-mode diagnostic flags.** The MoGe ROE / DIODE outdoor /
   ETH3D chamfer gaps this session surfaced took individual
   investigations. A `plumbline reproduce --diagnose` that dumps
   per-sample inlier counts, per-protocol alignment residuals, and a
   "paper-row checklist" would cut debug time on the next paper.

9. **DTU v0.1 gate.** vggt-paper-dtu-mvs YAML is ready with ICP +
   paper target chamfer=0.382. Data retrieval is slow but viable
   (SampleSet.zip 6.9 GB at ~680 KB/s ≈ 3h). Once extracted,
   dependent on the Tier-1 chamfer rewrite to actually hit paper.

### 2026-04-19 pivot: deprioritize Sintel + ScanNet (auth-gated)

Both were originally in v0.1's 3-dataset shortlist. We spent a GPU
session pinning 13 paper reproductions without either; neither has
its access email landed yet either. Rather than block on the emails,
promote public substitutes:

| Dropped (auth-gated) | Substitute (public) | What it covers |
|---|---|---|
| ScanNet-1500 pose | **Co3Dv2** | two-view + multi-view pose paper rows (VGGT Table 1, DUSt3R, MASt3R) |
| ScanNet-1500 pose | **7Scenes** | relocalization paper rows (MASt3R) |
| Sintel depth | **GSO** (Google Scanned Objects) | synthetic clean-GT slot (MoGe Table 2) |
| Sintel depth | **iBims-1** | high-fidelity indoor synthetic (MoGe Table 1/2) |

Sintel and ScanNet loaders stay in the repo (they work, just blocked
on data access). When either email lands, the infrastructure is
ready; otherwise, **a good benchmark is achievable without either**
using the substitutes above.

### Tier 4 — parking lot

- **Novel-view synthesis evaluation** (PSNR/SSIM/LPIPS).
- **Point tracking evaluation** (TAP-Vid, BADJA).
- **Uncertainty calibration metrics.**
- **Failure-case browser web UI.**
- **Nightly CI running the full suite.**
- **Hosted leaderboard site.**
- **Additional datasets**: TUM-dynamics, Replica, Tanks & Temples,
  **NRGBD** (Neural RGB-D scene dataset — Azinović et al. 2022, indoor
  RGB-D used by recent SLAM / reconstruction papers; worth considering
  once we're testing SLAM-adjacent benchmarks).
- **HDR / linear-color evaluation path** (leverages framewright).
- **Distributed eval across multiple GPUs.**

### Tier 5 — model-roster tracker

Full current + planned adapter roster so no one loses track.

| Adapter | Status | Paper rows available |
|---|---|---|
| DA-V2 (S/B/L + metric I/O variants) | **shipped** | NYU S/B/L ✅, DIODE-indoor ✅, KITTI TODO |
| Metric3D-v2 (S/L/Giant) | **shipped** | NYU L/G ✅, KITTI TODO |
| MASt3R (2-view PairViewer) | **shipped** | ScanNet-1500 blocked on data |
| VGGT (1B) | **shipped** | ETH3D / DTU chamfer — protocol gap; pose tables TODO |
| DA3 (Large-1.1) | **shipped** | NYU ✅, multi-view TODO |
| MoGe (v1 + v2 variants) | **shipped** | NYU ✅ under ROE; DIODE in progress |
| Marigold (v1-1, v1-0) | **shipped 2026-04-19** | NYU in progress; KITTI TODO |
| **Depth Pro** (Apple 2024) | planned (Tier 2) | NYU / KITTI / DIODE paper rows |
| **GeoWizard** (depth + normals) | planned (Tier 2) | NYU / KITTI / ETH3D |
| **DepthFM** (flow-matching) | planned (Tier 2) | NYU / KITTI |
| **π³** (Pi-Cubed) | planned (Tier 2) | multi-view depth benchmarks |
| **CUT3R** (continuous 3D tokenizer) | planned (Tier 2) | sequential multi-view |
| **Fast3R** (Meta) | planned (Tier 2) | many-view 3D reconstruction |
| **FLARE** | planned (Tier 2) | fast multi-view 3D (confirm repo) |
| **MapAnything** | planned (Tier 2) | 3D recon w/ map prior |
| **MonST3R** (dynamic DUSt3R) | planned (Tier 3) | dynamic-scene 3D |

#### Dataset roster (current + Tier-2 planned)

| Dataset | Status | Primary use |
|---|---|---|
| NYUv2 | **shipped** | indoor mono-depth (7 paper-matches) |
| KITTI | loader **shipped**, data partial | outdoor mono-depth (DA-V2/MoGe/Metric3D/Depth Pro paper rows) |
| DIODE | **shipped**, seg-mask integration in flight | mixed indoor/outdoor dense-LiDAR (MoGe / DA-V2 paper rows) |
| ETH3D high-res | **shipped** | multi-view chamfer (VGGT / DA3 / MASt3R) |
| DTU MVS | **shipped**, GT download in flight | v0.1 paper-match gate (VGGT Table 2) |
| **Co3Dv2** | loader **shipped** (2026-04-19) | pose paper rows (VGGT Table 1, DUSt3R, MASt3R) |
| **GSO** | loader **shipped** (2026-04-19), 1030 objects on disk | synthetic clean-GT (MoGe Table 1/2) |
| **7Scenes** | planned (Tier 3) | classical relocalization |
| **iBims-1** | planned (Tier 3) | high-fidelity indoor synthetic |
| Sintel | **deprioritized 2026-04-19** | auth-gated; substituted by GSO / iBims-1 |
| ScanNet + ScanNet-1500 | **deprioritized 2026-04-19** | loaders wired, data auth-gated; Co3Dv2 covers pose protocol |

### What's NOT in v0.2 (and why)

- **Training / fine-tuning**: still v0.1's non-goal. Plumbline evaluates.
- **Novel metrics**: only what papers report.
- **Web UI / leaderboard**: post v0.2.

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
