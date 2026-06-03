# DAGE — additional validation opportunities

Scoping pass over the DAGE paper (Ngo et al. 2026, arXiv:2603.03744, CVPR 2026)
to find **datasets and method rows we could reproduce** beyond what plumbline
already validates.

## What's already validated

| Cell | Table | Status |
|------|-------|--------|
| `dage-sintel-pose` (DAGE, Sintel, ATE 0.132) | Table 4 | ✅ MATCH — ATE 0.1417 (+7.3 %), companions within ~2 % |
| `cut3r-sintel-pose-dage` (CUT3R baseline, Sintel, ATE 0.217) | Table 4 | ✅ MATCH — ATE 0.2138 (−1.5 %) |

The DAGE adapter (`src/plumbline/models/dage.py`) currently wires **only the
pose task** (`camera_poses → extrinsics`). Depth / point-map outputs are
deferred (returned at the internal 252px lr resolution; need unscaling).

## The DAGE paper's tables

| Table | Task | Datasets | Metric(s) |
|-------|------|----------|-----------|
| 1 | Video geometry (point-map) | GMU, Monkaa, Sintel, ScanNet, KITTI, UrbanSyn, Unreal4K, Diode | Rel^p↓, δ^p↑ |
| 2 | Depth boundary sharpness | Monkaa, Sintel, UrbanSyn, Unreal4K | F1↑, C_PDBE↓ |
| 3 | Multi-view reconstruction | 7-Scenes, NRGBD (sparse/dense/metric) | Acc↓, Comp↓, NC↑ |
| 4 | Camera pose | **Sintel, TUM-Dynamics, ScanNet** | ATE↓, RPE-trans↓, RPE-rot↓ |
| 5 | Runtime | — | FPS, mem |
| 6/7 | Ablations | NRGBD, Sintel | — |

### Table 4 full rows (camera pose, ATE / RPE-trans / RPE-rot°)

| Method | Sintel | TUM-Dyn | ScanNet | In plumbline? |
|--------|--------|---------|---------|---------------|
| Fast3R | 0.371 / 0.298 / 13.75 | 0.090 / 0.101 / 1.425 | 0.155 / 0.123 / 3.491 | ✗ no adapter |
| CUT3R  | 0.217 / 0.070 / 0.636 | 0.047 / 0.015 / 0.451 | 0.094 / 0.022 / 0.629 | ✅ cut3r (Sintel done) |
| FLARE  | 0.207 / 0.090 / 3.015 | 0.026 / 0.013 / 0.475 | 0.064 / 0.023 / 0.971 | ✗ no adapter |
| VGGT   | 0.167 / 0.062 / 0.491 | 0.012 / 0.010 / 0.311 | 0.035 / 0.015 / 0.382 | ✅ vggt (bf16-blocked on 1080Ti) |
| Pi3    | 0.074 / 0.040 / 0.282 | 0.014 / 0.009 / 0.312 | 0.031 / 0.013 / 0.347 | ✗ adapter was removed |
| VGGT (252px) | 0.228 / 0.095 / 1.03 | 0.053 / 0.028 / 0.652 | 0.109 / 0.039 / 1.357 | ✅ vggt (bf16-blocked) |
| Pi3 (252px)  | 0.153 / 0.088 / 0.684 | 0.025 / 0.019 / 0.370 | 0.045 / 0.017 / 0.438 | ✗ adapter removed |
| **DAGE** | **0.132 / 0.051 / 0.406** | **0.014 / 0.010 / 0.323** | **0.031 / 0.014 / 0.389** | ✅ dage (Sintel done) |

## Key enabler: Table 4 reuses the MonST3R relpose harness

DAGE's `eval_pose_dage.py` is built on the **same MonST3R pose-eval lineage**
plumbline already runs for `monst3r-sintel-pose` / `dage-sintel-pose`. From the
vendored `monst3r/dust3r/eval_metadata.py`, all three Table-4 columns are the
**same protocol** — Sim(3)-aligned (Umeyama, `correct_scale=True`) TUM-RGBD
ATE/RPE-RMSE via `evo`, over **90-frame subsampled clips**:

- **Sintel** — 14 dynamic-final clips (already wired).
- **ScanNet** — `<scene>/color_90/*.jpg` + `pose_90.txt` (replica traj format),
  iterates all staged scenes.
- **TUM-Dyn** — `<seq>/rgb_90/*.png` + `groundtruth_90.txt` (tum traj format).

plumbline already emits exactly the three metrics this needs
(`trajectory_ate_rmse_sim3`, `trajectory_rpe_trans_rmse`,
`trajectory_rpe_rot_deg_rmse` — see `metrics/pose.py`). So the **metric +
apparatus are done**; what's missing for the new columns is data staging + a
loader that emits a full-clip trajectory.

## Prioritized opportunities

### Tier 1 — new pose columns, models already fit the 1080Ti

The pose apparatus is wired and DAGE/CUT3R are matched on Sintel; both are
feed-forward and fit 11 GB. Adding TUM-Dyn and ScanNet columns is the
highest-leverage, lowest-risk extension.

1. **DAGE on ScanNet pose** — target ATE 0.031 / 0.014 / 0.389.
   - `scannet` loader exists but reads raw `scans_test/<scene>/color/`. Needs a
     **video-pose mode**: pick the MonST3R ScanNet scene set, subsample to the
     90-frame clips, emit clip + GT trajectory (`world_from_camera`). Either add
     the mode to `datasets/scannet.py` or stage the MonST3R-preprocessed
     `color_90`/`pose_90` layout and add a thin loader.
   - Confirm the exact scene list against MonST3R's published ScanNet pose set
     (the number must match 0.031).
2. **DAGE on TUM-Dynamics pose** — target ATE 0.014 / 0.010 / 0.323.
   - **New loader needed** (`tum-dynamics`). TUM RGB-D is public + small. Stage
     the freiburg3 dynamic sequences MonST3R uses, subsample to `rgb_90` +
     `groundtruth_90.txt`, emit clip + TUM-format GT trajectory.
3. **CUT3R on ScanNet + TUM-Dyn** (baseline rows: ScanNet 0.094, TUM 0.047) —
   free once (1)/(2)'s loaders land; `cut3r` already runs on the 1080Ti
   (`cut3r-sintel-pose-dage` ✅). Same model/protocol cross-check as the Sintel
   CUT3R cell.

### Tier 1.5 — needs a bigger (bf16) GPU, adapter already exists

4. **VGGT on Sintel / TUM-Dyn / ScanNet pose**, both full-res and **252px**
   (Table 4: 0.167 / 0.228 Sintel, etc.). `vggt` adapter exists; pose is
   bf16-blocked on the 1080Ti (per project notes, needs a 3090/H100). The 252px
   rows are the apples-to-apples comparison DAGE highlights, so they're the most
   interesting baseline to land. Sintel loader is ready today; TUM/ScanNet share
   Tier-1's loader work.

### Tier 2 — needs new DAGE adapter output + metric work

5. **DAGE multi-view reconstruction (Table 3)** on **7-Scenes** — the `7scenes`
   loader already carries per-frame depth + `world_from_camera` pose. Needs:
   (a) DAGE point-map output wired in the adapter (currently pose-only), and
   (b) the Acc/Comp/**NC (normal-consistency)** chamfer-style metric — plumbline
   has chamfer (DTU/ETH3D) but NC may need adding. NRGBD (other Table-3 column)
   has no loader.
6. **DAGE point-map / video geometry (Table 1)** on **Sintel / ScanNet / KITTI /
   Diode** (all have loaders). Needs the DAGE point-map output path **plus** the
   `Rel^p` / `δ^p` point-map-space relative-error metric (non-standard; not
   currently in plumbline). Bigger lift; defer behind Table 3.

### Not feasible without new model adapters

- **Fast3R, FLARE, Pi3** baseline rows — no adapters (Pi3's was removed). Out of
  scope unless those models are re-added.
- **Table 2** (boundary F1 / C_PDBE) — specialized sharpness metrics plumbline
  doesn't implement; low priority.

## Recommended next step

Land the **TUM-Dynamics loader + ScanNet video-pose mode** (Tier 1) — one focused
piece of loader work unlocks **six new verified cells** at once (DAGE ×2 targets,
CUT3R ×2 baselines, plus the Sim(3) apparatus is already proven). Validate the
ScanNet/TUM scene lists against MonST3R's published sets first so the targets are
exactly comparable.

## Status (2026-06-03) — Tier 1 loaders landed

Implemented the Tier-1 loader work; the scene/sequence lists were confirmed
against DAGE's own `evaluation/relpose/metadata.py` (which reuses MonST3R's
`prepare_tum.py` / `prepare_scannet.py` and `download_*.sh` verbatim):

- **`tum-dynamics` loader** (`src/plumbline/datasets/tum_dynamics.py`) — reads the
  8 freiburg3 dynamic sequences, replicates MonST3R's prep at read time (associate
  rgb↔groundtruth, first-90-at-stride-3), emits one trajectory Sample/sequence.
  Staging: `scripts/stage_tum_dynamics.py` (public `.tgz`, no ToS). Unit-tested.
- **`scannet-video-pose` loader** (`src/plumbline/datasets/scannet_video_pose.py`)
  — reads the MonST3R `color_90`/`pose_90.txt` layout, drops dropped-tracker
  frames, emits one trajectory Sample/scene. Unit-tested. **Data-blocked** on
  ToS-gated raw ScanNet (scenes `scene0707_00..scene0806_00`).
- **Configs:** `reproductions/dage_tum_pose.yaml` (ATE 0.014),
  `cut3r_tum_pose_dage.yaml` (0.047), `dage_scannet_pose.yaml` (0.031),
  `cut3r_scannet_pose_dage.yaml` (0.094). All four added to `gpu_queue.yaml`
  (TUM ×2 `pending` / 1080Ti-runnable; ScanNet ×2 `blocked` on data).

**Ran 2026-06-03 (GTX 1080Ti, 8/8 sequences, member-selective staging ~366 MB):**

- **`dage-tum-pose` ✅ MATCH** — ATE **0.0136** vs 0.014 (−2.9 %); companions
  RPE-trans 0.0104 vs 0.010, RPE-rot 0.3213 vs 0.323 — all within ~4 %. **New
  verified pose cell** (the DAGE pose axis's second dataset column after Sintel).
- **`cut3r-tum-pose-dage` ℹ️ informational** — ATE 0.0362 vs 0.047 (−23 %), but
  companions near-exact: RPE-trans 0.0150 vs 0.015 (exact), RPE-rot 0.4486 vs
  0.451 (−0.5 %). Baseline cross-measurement (DAGE ran competitors at 518 px vs
  CUT3R's 512 default); the tight RPE agreement confirms the apparatus.

Gotcha fixed along the way: both loaders now key their manifest cache on the set
of *present* sequences, so staging more data after a first partial scan
invalidates the cache (a stale 1-sequence manifest first gave ATE 0.0099 — the
single-sequence value, not the 8-sequence mean).

**Next:** ScanNet stays blocked until raw ScanNet is staged + run through
`prepare_scannet.py` (`scannet-video-pose` loader + configs are code-ready).
