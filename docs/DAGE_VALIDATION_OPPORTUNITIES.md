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

## Methods DAGE compares against (can we add baselines?)

Every baseline row across DAGE's tables, vs whether plumbline has the adapter:

| Method | DAGE tables | plumbline adapter? | Runnable today? |
|--------|-------------|--------------------|-----------------|
| CUT3R | 1,2,3,4 | ✅ `cut3r` | ✅ pose done (Sintel+TUM); 1080Ti |
| VGGT | 1,2,3,4 | ✅ `vggt` | ⚠️ pose: bf16 (→H100) **and** `max_views=49` < 90-frame clips |
| Pi3 (π³) | 1,2,3,4 | ❌ removed (63ada6a, "no anchor") | depth/chamfer: 1080Ti fp16; pose: `max_views=16` caps it |
| DepthPro | 1,2 | ✅ `depth-pro` | needs Rel^p / F1 metrics (Table 1/2) |
| MoGe | 1 | ✅ `moge` | needs Rel^p metric |
| MoGe-2 | 1,2 | ✅ `moge` (moge2 cfgs) | needs Rel^p / F1 metrics |
| DUSt3R | related work | ✅ `dust3r` | — |
| DepthAnything-V2 | related work | ✅ `depth-anything-v2` | — |
| Fast3R | 3,4 | ❌ no adapter | new adapter |
| FLARE | 3,4 | ❌ no adapter | new adapter |
| MapAnything | 3 | ❌ no adapter | new adapter |
| GeoCrafter | 1,2 | ❌ no adapter | new adapter |

**Takeaways:**
- **7 of DAGE's baselines already have adapters** (CUT3R, VGGT, Pi3-ish, DepthPro,
  MoGe, MoGe-2, DUSt3R, DA-V2) — but cross-validating their DAGE rows is gated by
  *metrics* (Rel^p, F1/C_PDBE, NC) and *DAGE's own depth/pointmap output* not yet
  wired, not by the adapters.
- **Pose (Table 4) is the only fully-wired metric**, and there CUT3R is the only
  present baseline that's also trajectory-capable (done). VGGT (`max_views=49`) and
  Pi3 (`max_views=16`) can't hold the 90-frame clips without chunking, and VGGT
  needs bf16.
- **Pi3 is re-introducible with a fresh justification**: it was removed for "no
  anchor", and DAGE now *supplies* anchors (Table 4 pose 0.074/0.014/0.031, Table 1/3
  rows). It runs on the 1080Ti at fp16 for depth/chamfer (Table 1/3), though its
  pose cells are view-capped.
- New adapters (Fast3R, FLARE, MapAnything, GeoCrafter) are larger lifts and less
  standard.

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

## Status (2026-06-05) — re-scope after the model-roster + storage work

Three session changes materially update the DAGE opportunity landscape:

**1. Pi3 is back (vendored, PR #60) — DAGE's strongest pose baseline is now
validatable.** The 2026-06-03 scope said "Pi3 adapter removed → can't validate its
rows." That's stale. In Table 4 Pi3 actually *beats* DAGE on pose (Sintel
0.074 vs 0.132; ScanNet 0.031 vs 0.031 tie), so its rows are worth landing.
- **Full-res Pi3 (~505px, PIXEL_LIMIT=255000 = DAGE's full-res row, ATE 0.074)
  OOMs the 1080 Ti** on the 50-frame Sintel clips (Pi3 has no lr-downscale
  stream like DAGE's `lr_max_size:252`).
- Added a **`pixel_limit` kwarg** (PR `feat/pi3-pixel-limit`) so Pi3 can run at
  **252px → the DAGE "Pi3 (252px)" row, Sintel ATE 0.153** — ~4× less VRAM, fits.
  `pi3-sintel-pose` @ 252px running 2026-06-05 (result appended below).
- TUM-Dyn/ScanNet Pi3 rows stay capped: those clips are 90 frames > Pi3
  `max_views=64`; full-res also OOMs. Pi3 full-res anywhere needs an Ampere/
  bigger-VRAM box.

**2. Full VGGT-family coverage.** vggt + **streamvggt** + **vggt-omega** all have
adapters now (roster PRs #56–#62). All are DAGE-comparable (VGGT is a Table-1..4
baseline), but all are **bf16/Ampere-blocked on the 1080 Ti** — same wall as
before, just more models behind it.

**3. Storage fixed — ScanNet's blocker is ToS, not disk.** The box now has 3.3 TB
free on /data3 (eval data + caches relocated there). So the ScanNet pose cells
(DAGE/CUT3R/Pi3 ScanNet, Table 4) are **gated on the ToS-signed raw ScanNet
download**, not capacity — the `scannet-video-pose` loader + configs are
code-ready (2026-06-03). ETH3D etc. that were disk-blocked are now stageable.

### Updated runnable-today matrix (1080 Ti)
| cell | status |
|---|---|
| DAGE Sintel/TUM pose | ✅ verified |
| CUT3R Sintel/TUM pose | ✅ done (Sintel ✅, TUM ℹ) |
| **Pi3 Sintel pose @ 252px (target 0.153)** | ▶ running 2026-06-05 |
| Pi3 Sintel pose full-res (0.074) | ⛔ OOM (needs Ampere) |
| DAGE/CUT3R/Pi3 ScanNet pose | ⛔ ToS-gated ScanNet data |
| VGGT/StreamVGGT/VGGT-Ω pose | ⛔ bf16 (needs H100) |
| DAGE Table 3 recon (7-Scenes) | ⛔ needs DAGE point-map output + NC metric |

### Pi3-Sintel-pose result (2026-06-06)

`pi3-sintel-pose` ran on the 1080 Ti (variant=pi3, 252px/pixel_limit=63504, fp32,
14/14 clips): **ATE 0.0826, RPE-t 0.0491, RPE-r 0.3397** — INFORMATIONAL. It
**brackets DAGE's two Pi3 rows** (full-res 0.074 / 252px 0.153), landing closest
to full-res (+11.6%). Pi3 pose is strongly resolution-sensitive; pixel_limit=63504
(~385px long-edge) is between DAGE's two resolution points. Wiring fixed en route:
`evo` was missing (trajectory metrics silently skip without it) and the adapter
defaults to `pi3x` (DAGE tables plain `pi3`). To land DAGE's exact points: full-res
0.074 needs an Ampere/bigger-VRAM box (OOMs the 1080 Ti); 252px 0.153 needs
pixel_limit≈27k. **Net: Pi3 confirmed as a runnable DAGE baseline — the first new
DAGE-axis cell since TUM, informational pending the exact-resolution match.**
