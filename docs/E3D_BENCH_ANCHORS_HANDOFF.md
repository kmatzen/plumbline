# Handoff — E3D-Bench as a verified anchor for new plumbline cells

**Context.** Scoped while looking to add a **7-Scenes pose** reproduction (the
`7scenes` loader is built + fixture-validated but unwired — no reproduction
references it). Finding below redirects that idea.

## 7-Scenes pairwise pose has no clean anchor — don't chase it

The `7scenes` loader docstring already warns that pairwise pose AUC on 7-Scenes
has no canonical sampler (VGGT/CUT3R/Spann3R/Fast3R each define their own). The
obvious candidate for a *common* protocol — **E3D-Bench** (arXiv:2506.01933,
`github.com/VITA-Group/E3D-Bench`) — **does not evaluate 7-Scenes for pose**.
Per its Table 4 (Multi-view Relative Pose Estimation), the pose datasets are
grouped CO3Dv2 · ADT/TUM-Dyn · KITTI-Odometry · Bonn/Sintel/RGBD ·
ACID/Syndrone · ULTRRA. **7-Scenes appears only in the reconstruction table
(Table 5).** So a 7-Scenes pose cell would still be a per-paper, idiosyncratic
target — skip it.

## E3D-Bench *is* a strong anchor for other new cells

E3D-Bench is a benchmark paper (verified_pdf-eligible) that evaluates models
plumbline already ships — **CUT3R, VGGT, MonST3R, DUSt3R, MASt3R** — under one
documented protocol. Two task tables are directly usable:

### Table 4 — Multi-view Relative Pose Estimation (ATE ↓, RPE-trans ↓, RPE-rot ↓)
- Sim(3) Umeyama alignment to GT trajectory, per-scene-group means.
- plumbline **already computes trajectory ATE/RPE** (MonST3R-Sintel pose,
  `project_monst3r_sintel_pose_v3`) — the metric family exists.
- New cells available without new metrics: e.g. **VGGT / CUT3R / MonST3R on
  KITTI-Odometry, Bonn/Sintel** (ATE/RPE). Bonn + Sintel
  loaders already exist; KITTI does too.
- Gotcha: E3D-Bench reports **per-scene-group** means (e.g. "Bonn & Sintel &
  RGBD" pooled), so a faithful cell must match its grouping + frame sampler.

### Table 5 — Sparse/Dense 3D Reconstruction (Acc ↓, Comp ↓, NC ↑) on 7-Scenes (+DTU/NRGBD/TUM)
- This is where the **`7scenes` loader finally gets used**.
- Umeyama-aligned, official masks; two settings: **Extremely-Sparse (2–5 imgs)**
  and **Dense (10–50 imgs)** per scene.
- plumbline has **chamfer** (≈ Acc+Comp) but **no Normal-Consistency (NC)**
  metric — NC would need implementing in `metrics/pointmap.py` (per-point normal
  cosine vs GT surface) to score the full Table-5 triple.
- Most tractable first cell: **CUT3R 7-Scenes (Dense)** — CUT3R is recurrent /
  low-memory (E3D-Bench Fig. 2 shows it among the lightest), so it fits the
  **11 GB 1080Ti** for 10–50-frame sequences, unlike VGGT-1B which OOMs at high
  view counts. Score Acc/Comp now (chamfer-style); add NC when implemented.

## Concrete next steps (in order)
1. Pull the exact protocol from `VITA-Group/E3D-Bench`: per-table frame samplers
   (which scenes, how many frames, stride), alignment (Sim3 Umeyama), and mask
   handling. The repo's eval scripts are authoritative; the paper text is not
   enough.
2. **Verify the target numbers against the PDF** (Table 4 / Table 5), table +
   column + row, before pinning any `paper_reference` (per project policy).
   Exact small-font values must be read from the camera-ready, not inferred.
3. Stage data (all free / already-staged): 7-Scenes (Microsoft, no login),
   Bonn (have), Sintel (have).
4. If pursuing Table 5: implement an **NC metric** in `metrics/pointmap.py`
   (+ test) so the cell can score the full Acc/Comp/NC triple.
5. Wire protocol + reproduction YAMLs (one per cell), matching E3D-Bench's
   per-scene-group aggregation.

## Recommendation
Start with **Table 4 pose cells** (no new metric needed — ATE/RPE already exist):
the cheapest path to new verified coverage. `CUT3R`/`MonST3R` on
Bonn/Sintel is the first candidate. Defer Table 5 / 7-Scenes until the NC metric
is worth building.
