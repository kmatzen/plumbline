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

### Laptop queue

Empty. All paper-match-blocking laptop work has landed.

### Landed, awaiting next-GPU verification

- **D20 · scene-aggregation chamfer perf** — per-sample ICP lifted to
  once-per-scene on the fused+voxel-downsampled prediction cloud
  (`src/plumbline/runner.py` scene-merge loop). Unblocks D3/D4
  verification that timed out on 2026-04-21.
- **D19 · MoGe-DIODE disparity clamp** — new `scale_shift_clamped`
  alignment mode applies MoGe's `1/gt.max()` disparity floor
  (`src/plumbline/metrics/alignment.py`); `moge_vitl_diode_{both,indoor}.yaml`
  opt in. Prediction cache-hits — next-session re-scores in seconds.
- **D8 / D9 / D18 · KITTI MoGe-eval protocol** — new
  `KITTIMogeEvalLoader` + `kitti_moge_eval` protocol
  (`src/plumbline/datasets/kitti.py`, `protocols/kitti_moge_eval.yaml`).
  `moge-vitl-kitti` + `marigold-v1-1-kitti` + `geowizard-kitti` opt in.
  Needs the HF-bundle KITTI data staged to `$KITTI_MOGE_ROOT`:
  ```
  hf download Ruicheng/monocular-geometry-evaluation \
      KITTI.zip --repo-type dataset --local-dir <tmp>
  unzip <tmp>/KITTI.zip -d $KITTI_MOGE_ROOT
  ```

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
2. **Never invent paper numbers.** `source_confidence: verified_pdf`
   is a contract with the paper's PDF.
3. **Never delete S3 cache contents.** Shared across future sessions.
4. **Never use credentials other than the session token.** No
   copying long-lived keys onto the rental box.
5. **Never bypass hooks with `--no-verify`.**

Source edits, commits, pushes, and PRs from the rental/GPU box are
fine — treat the box like any other dev environment. The constraints
above cover the real concerns (YAML/citation integrity, secrets,
shared state); workflow hygiene is not worth its own gate.

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
