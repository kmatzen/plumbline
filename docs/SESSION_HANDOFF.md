# Session handoff — pick up here

Last updated: 2026-04-22. This doc is the one-pager for the next GPU
session (human or Claude-Code agent). Everything else branches from here.

## Where we are

The 2026-04-21 rental run exercised the full paper-match queue against
real weights + real data. Results:

- **12 / 20 MATCH** — paper-match claim validated.
- **7 OFF-PAPER** — observed but off; each root-caused in
  `docs/DISCREPANCIES.md` (most are protocol/loader deltas, not adapter
  bugs).
- **1 FAILED** — `vggt-eth3d-multiscene-chamfer`; fix landed, verification
  timed out (see D20 below).

Full per-row numbers + env deviations + off-paper diagnoses:
`docs/runs/20260421.md`. Open issues catalog: `docs/DISCREPANCIES.md`.
Status matrix: `REPRODUCTIONS.md`.

## What's next

`docs/DISCREPANCIES.md § Priorities for the next session` lists open work
in cheap-first order. Summary:

### Highest-leverage laptop prep (not yet implemented)

These are blocking future GPU verification — doing them on the laptop
before the next rental avoids burning a second timeout.

1. **D20 · scene-aggregation chamfer perf.** Lift per-sample ICP in
   `src/plumbline/runner.py:465 _aligned_point_map` to once-per-scene on
   the fused+voxel-downsampled prediction cloud. Blocks verification of
   D3 (VGGT-DTU) and D4 (VGGT-ETH3D) — the 2026-04-21 run timed out in
   this path. 1–2 h CPU-only work.
2. **D19 · MoGe-DIODE per-sample disparity clamp.** Add MoGe's
   `clamp_min(1 / gt_depth.max())` to the predicted disparity before
   inversion (ref: `moge/test/metrics.py:210`). Median already lands on
   paper; mean is blown up by a handful of outdoor outliers. ~1 h.
3. **D8/D9/D18 · KITTIMogeEvalLoader + kitti_moge_eval protocol.**
   MoGe/Marigold/GeoWizard all miss KITTI by 10–35 % under plumbline's
   Monodepth2-Eigen + Garg protocol; they publish under MoGe's bespoke
   750×375 center-warp eval. One loader + protocol closes all three.
   4–6 h.

### Verify-on-next-GPU (fixes already on `main`)

These just need a clean GPU run to close out:

- **D3 · VGGT-DTU chamfer** — protocol fix (aggregation=scene, 1 cm
  voxel) landed in commits `ad924e9` + `2c140b3`. Blocked on D20.
- **D4 · VGGT-ETH3D multiscene chamfer** — scan_clean GT pushed to S3
  (commit `f89cdc4`). Blocked on D20.
- **Cleanup batch** (moge-diode-both under drop_max_depth, moge-kitti +
  marigold-kitti re-scored under Garg) — re-runs will cache-hit on
  prediction; metrics-only.

## GPU-session bring-up

Unchanged from the prior session:

- Human: `GPU_RUNBOOK.md`
- Claude-Code agent: `docs/AGENT_GPU_RUNBOOK.md`
- S3 cache layout: 7,287 objects / 54 GB at `s3://plumbline-bench/`
  (datasets 12 GB + hf-cache 35 GB + torch-hub-cache 7 GB + predictions
  49.7 GB). `scripts/stage_all_data.sh` syncs it all.
- Session token: `scripts/gpu_box_session_token.sh` on laptop → paste
  to rental box.

## Hard constraints (load-bearing — do not relax)

1. **Never modify reproduction YAMLs.** If a paper number doesn't
   reproduce, that's a finding, not a parameter to tune.
2. **Never commit, push, or open PRs from the rental box.** Code
   changes come from the laptop.
3. **Never invent paper numbers.** `source_confidence: verified_pdf`
   is a contract with the paper's PDF.
4. **Never delete S3 cache contents.** Shared across future sessions.
5. **Never use credentials other than the session token.** No
   copying long-lived keys onto the rental box.
6. **Never bypass hooks with `--no-verify`.**

The 2026-04-20 audit codifying these rules is `reproductions/AUDIT.md`;
long-term agent memory has `feedback_paper_citations.md`.

## Deferred to v0.2

Documented in `plan.md § 10` but not on the v0.1 critical path:

- Depth Pro paper rows (paper evals Sun-RGBD / ETH3D / Middlebury /
  etc. — need new loaders).
- Pi3 verified-PDF pinning (multi-view chamfer units unclear until
  first-run observation).
- 7-Scenes + Co3Dv2 pose benchmarks (loaders wired, data not staged).
- MoGe-2 / MASt3R / Pi3 HF weight staging.
- DIODE outdoor protocol polish (D19 mean-collapse) + D8/D9 MoGe-KITTI
  structural loader.
- Sintel (auth-gated; deprioritized on 2026-04-19 pivot).
