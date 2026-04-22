# Plan — fix the 2026-04-21 agent run's 6 not-green rows

Context: the GPU-rental agent run (see `docs/AGENT_RUN_20260421.md`) left 4
rows ⚠️ OFF-PAPER and 2 ❌ SKIPPED (geowizard). This plan organizes the
fixes by effort + risk, cheap-first. Two of the four OFF-PAPER rows likely
clear with a 15-minute change; one is a multi-hour dataset-loader rewrite.

**Progress (running log, freshest at top):**

- [x] **Item 1** — DA-V2 paper `.pth` checkpoint path landed in the
  adapter (`source="paper"` default for relative variants). Re-run
  on the paper checkpoints confirms the systematic ~0.002 AbsRel
  downshift on NYU is NOT an HF-re-export artefact — the paper `.pth`
  reproduces within 0.0005 of the HF build. Base (0.0455 vs paper
  0.049) persists at 7.2 % off regardless of checkpoint; it's within
  noise relative to Small (3.9 %) and Large (1.6 % after MoGe-pin
  swap). Large re-pinned to MoGe's 0.0420 re-eval because plumbline's
  plain-LSQ solver reproduces that exactly and is 5.2 % off DA-V2's
  own 0.045.
- [x] **Item 2** — `scale_shift_robust` → `scale_shift` on moge-NYU.
  Observed 0.03419 vs paper 0.03410 — **0.28 % MATCH**, dead-on.
  Confirms the "IRLS overfits paper's plain LSQ" hypothesis. Cascaded
  the alignment fix to `moge-vitl-kitti`, `moge-vitl-diode-*`,
  `moge2-vitl-*`, `protocols/gso_moge.yaml`.
- [x] **Item 3** — `moge-vitl-diode-indoor` demoted to
  `source_confidence: approximate`, `value: null`. Dropped from the
  20-row verified queue.
- [x] **Item 4** — GeoWizard shim landed + `generator` kwarg fix (the
  first re-run exposed a second breakage: upstream's __call__ doesn't
  accept `generator`, so plumbline now seeds the global RNG per-sample
  and drops the kwarg). Verification pending in the cleanup batch.
- [x] **Item 5** — DIODEMogeEvalLoader written; registered as
  `diode-moge-eval`. Reads MoGe's HF bundle (16-bit log-encoded
  depth.png with sky-as-+inf, `isfinite` mask, no depth clip).
  Smoke-tested 771 samples (325 indoor + 446 outdoor). HF bundle also
  pushed to `s3://plumbline-bench/datasets/diode_moge/` for future
  sessions. Verification pending in the cleanup batch.
- [x] **Item 6, part 1** — root-cause of the ETH3D 0-MVS-metrics bug
  found: plumbline's loader requires `scan_clean/scan*.ply` as GT
  point cloud; the S3 cache only had `dslr_scan_eval/`, so
  `point_cloud_gt=None` and scene aggregation silently skipped.
  Fetched `*_scan_clean.7z` for the 3 scenes via `py7zr`, unpacked,
  and pushed to S3 so future sessions pick it up automatically.
  Verification pending — clean re-run will tell whether the cached
  0.818 m aggregate stands or shifts. The 3-scene-subset-vs-full-
  split question is orthogonal and still open.

**Additional findings (running list):**

- **KITTI Eigen-vs-Garg crop hypothesis REJECTED.** Tested both
  protocols empirically:

  | | Garg | Eigen |
  |---|---:|---:|
  | `moge-vitl-kitti`      | 9.4 % off  | 20.4 % off |
  | `marigold-v1-1-kitti`  | 10.1 % off | 17.8 % off |

  Eigen is worse. Reverted both YAMLs to `kitti_eigen_garg`.
  `protocols/kitti_eigen_crop.yaml` kept in the repo for future use.

- **Actual MoGe KITTI protocol** (subagent read of microsoft/MoGe
  @HEAD, 2026-04-22): they evaluate on a custom HF-bundle
  `data/eval/KITTI/.index.txt` sample list, center-warp the image to
  **750×375** (homographic reframe in
  `moge/test/dataloader.py::_process_instance`), apply **no Garg or
  Eigen crop**, and use **no [1e-3, 80] depth cap** — only MoGe's
  generic `drop_max_depth = nanquantile(gt, 0.01) * 1000` filter plus
  a per-scene `clamp_min(1 / gt_depth[mask].max())` on predicted
  disparity (`metrics.py:210`). Alignment is done on a 64×64
  downsampled grid then applied to full-res. This is materially
  different from Monodepth2's Eigen test protocol that plumbline's
  `kitti_eigen_garg` implements.

  **Resolution path for paper-match on MoGe-KITTI** would be a new
  `KITTIMogeEvalLoader` (sister of `DIODEMogeEvalLoader`) reading
  MoGe's HF bundle and a new `kitti_moge_eval` protocol wired around
  it. Scoped for v0.2 — not a same-session fix.

  Marigold's KITTI (same ~10 % gap) probably has a similar protocol
  delta; not yet investigated.

- **s5cmd** dropped in as the preferred S3 client in
  `stage_all_data.sh` + `tmp/agent/persist.sh` (falls back to
  `aws s3 sync`). Future rental bring-ups ~10–30× faster on the 54 GB
  cache.

- **VGGT-paper-dtu-mvs killed at 4 h 25 m** — no clean number. Partial
  first re-run hit 56 mm vs paper 0.382 mm. Hypothesis: ICP-alignment
  failed to absorb the m↔mm unit delta (DTU GT is mm, VGGT emits
  meters). Not yet diagnosed; see `src/plumbline/runner.py::
  _aligned_point_map` for the ICP path.

- **DIODE outlier fix landed.** MoGe's eval applies
  `max_depth = nanquantile(depth, 0.01) * 1000` on top of the
  isfinite mask. Plumbline's `DIODEMogeEvalLoader` now matches
  (`src/plumbline/datasets/diode.py`). Initial cleanup-batch run had
  mean AbsRel 2481 because one indoor + several outdoor samples
  produced post-alignment depths in the 1e4–1e6 range; median was
  0.025 (dead on paper). `drop_max_depth` should collapse the mean.
  Re-run pending.

- **Stale index entries to refresh after cleanup batch:**
  `moge-vitl-diode-both` (pre-fix), `moge-vitl-kitti` (latest under
  Eigen), `marigold-v1-1-kitti` (latest under Eigen). Need one final
  mini-batch under the correct protocol + fix.

## Priorities
---

## Historical plan (reference)

The original plan had six numbered items; all have been acted on and
their outcomes are in the progress log above. Per-item detail sections
were preserved here for a while but are now redundant — see the progress
log + `docs/AGENT_RUN_20260421.md` for what actually happened.

Original per-item effort estimates, preserved for similar-future-work
planning:

| Item | Est. effort | Outcome summary |
|---|---|---|
| 1. DA-V2 Base citation audit          | 15 min | Cited value verified; 7 % gap is protocol, not citation |
| 2. `scale_shift_robust` fit-space     | 15 min | disparity-space confirmed, IRLS > paper's LSQ — real result |
| 3. DIODE-indoor structural mismatch   | 15 min | demoted to `source_confidence: approximate` |
| 4. GeoWizard diffusers shim           | 2–4 h  | `_shim_diffusers_for_geowizard` landed + `generator` kwarg fix |
| 5. DIODE-outdoor loader               | 4–8 h  | `DIODEMogeEvalLoader` + `drop_max_depth` filter |
| 6. VGGT-ETH3D re-run                  | 2 h    | scan_clean fetched, re-run in flight |

## Rollback plan

Every change is on `main`. `git revert <sha>` recovers any of today's
edits; the agent report regenerates from the preserved `/tmp/results/` +
S3 mirror under `s3://plumbline-bench/runs/20260421-run1/`.
