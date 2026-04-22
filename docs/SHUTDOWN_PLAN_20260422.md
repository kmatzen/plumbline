# Shutdown plan — 2026-04-22 ~07:00 UTC

Target: clean shutdown of the rental box at 07:00 UTC, with all results
+ fixes + docs durably in S3 and on `main`. Drafted at 04:30 UTC
(2.5 h window).

## Timeline

| T+ | Clock | What | Duration |
|---|---|---|---|
| 0   | 04:30 | `geowizard-kitti` in flight (at 12.7 samples/min, 76 / 652) | +45 min |
| 45  | 05:15 | `vggt-eth3d-multiscene-chamfer` starts | +20 min |
| 65  | 05:35 | **Cleanup batch complete.** Kick off final mini-batch: | |
|     |       | — `moge-vitl-diode-both` (cache hit, drop_max_depth-only recompute) | ~1 min |
|     |       | — `moge-vitl-kitti` (cache hit, re-score under Garg) | ~1 min |
|     |       | — `marigold-v1-1-kitti` (cache hit, re-score under Garg) | ~1 min |
| 70  | 05:40 | Regenerate `docs/AGENT_RUN_20260421.md`, add `D17 geowizard NYU 10 %` to DISCREPANCIES | +10 min |
| 80  | 05:50 | **Optional: VGGT-DTU single-scan probe** (D3 diagnosis) | 30 min |
| 110 | 06:20 | Final incremental S3 sync (results + logs + predictions delta) | +10 min |
| 120 | 06:30 | Stop persister + progress monitor, final REPORT.md commit + push | +10 min |
| 130 | 06:40 | **20-min safety buffer** | |
| 150 | 07:00 | **Shutdown ready** | |

## Why the mini-batch is fast

All three rows hit the prediction cache:

- **moge-vitl-diode-both**: same config_hash; the `drop_max_depth` GT
  mask fix is loader-side, not model-side. Re-run re-reads cached
  MoGe predictions and re-computes metrics against the corrected GT
  mask. Total work: 771 samples × cheap mask + metric, ~30 s.
- **moge-vitl-kitti**: same config_hash; re-pointing the YAML from
  `kitti_eigen_crop` back to `kitti_eigen_garg` changes the evaluation
  crop but not the prediction. Metrics-only.
- **marigold-v1-1-kitti**: same as above. Marigold predictions are
  seeded, so even if re-inferred they'd be identical; but we already
  have them cached from the 04-21 Garg run.

## VGGT-DTU probe design (T+80 optional)

Goal: isolate why the partial run landed chamfer 56 mm vs paper
0.382 mm (147 × off, not ~1000 × m-vs-mm, not 1 × correct).

Probe: run 1 scan (scan1 — smallest, fastest) under three
`pointcloud_alignment` modes. Compare chamfer values:

| Mode | Expected behaviour | If chamfer lands... |
|---|---|---|
| `none` | no alignment; pred in m, GT in mm → ~1000 × off | we know the baseline m↔mm gap |
| `icp` | iterative scale + rotation + translation | what partial run did |
| `umeyama` | closed-form 7-DoF similarity | isolates ICP convergence vs solver |

The deltas between the three numbers tell us which transform is
misbehaving. Scan1 on VGGT 3090 inference: ~2 min; chamfer on
subsampled GT: ~5 min × 3 modes = 15-20 min total. Fits the 30-min
budget.

## What must be on S3 before shutdown

| Path | Contents | Already there? |
|---|---|---|
| `s3://plumbline-bench/runs/20260421-run1/results/` | All 25+ result .json | Persister (every 60 s) |
| `s3://plumbline-bench/runs/20260421-run1/logs/` | Per-repro stdout/stderr | Persister |
| `s3://plumbline-bench/predictions/` | 49.7 GB prediction cache | Yes (pushed earlier); **final sync at T+110 to pick up cleanup-batch additions** |
| `s3://plumbline-bench/datasets/kitti/depth_annotated/` | 14 GB Uhrig dense GT | Yes (pushed earlier) |
| `s3://plumbline-bench/datasets/eth3d/<scene>/scan_clean/` | MVS laser GT for 3 scenes | Yes (pushed earlier) |
| `s3://plumbline-bench/datasets/diode_moge/` | DIODE HF bundle | Yes (pushed earlier) |

## What must be on `main` before shutdown

Every commit has been auto-pushed by the agent; the final report + any
D17 update lands at T+120. Expected final diff vs `origin/main` start:

- `pyproject.toml`, `uv.lock` — torch downgrade
- `src/plumbline/runner.py` — progress bar
- `src/plumbline/models/{geowizard,depth_anything_v2}.py` — shim + paper-ckpt
- `src/plumbline/datasets/diode.py` — `DIODEMogeEvalLoader` + `drop_max_depth`
- `protocols/kitti_eigen_crop.yaml` (new), `protocols/gso_moge.yaml`, `protocols/diode_moge.yaml`
- Multiple `reproductions/*.yaml` — tolerance cap 0.05, citation fixes, alignment cascade
- `scripts/stage_all_data.sh` — s5cmd + DIODE_MOGE_ROOT
- `docs/AGENT_RUN_20260421.md`, `docs/AGENT_PLAN_20260421.md`,
  `docs/DISCREPANCIES.md`, `docs/ARCHITECTURE.md`,
  `docs/SHUTDOWN_PLAN_20260422.md` — new docs

## What I will NOT touch before shutdown

- `scripts/gpu_box_session_token.sh` — the laptop-side tool.
- The rental-box environment (deps, venv, /workspace/deps/* clones) —
  they're ephemeral.
- `~/.aws-creds` — session creds, not committed.
- The GitHub PR system — no PR opened; direct pushes to `main` were
  authorized this session.

## Post-shutdown resumption checklist

Any future session can pick up where this one left off by:

1. `scripts/gpu_box_session_token.sh` on laptop → paste into new box.
2. On new box: `git clone https://github.com/kmatzen/plumbline.git` +
   `uv sync --extra models`.
3. `scripts/stage_all_data.sh` — pulls datasets + hf-cache +
   **predictions** (new this session, saves re-inference).
4. If re-running any of the 20 queue targets, they'll cache-hit on the
   first prediction-cache lookup and finish in seconds.
5. If tackling D3/D8/D9 per `docs/DISCREPANCIES.md § Priorities`:
   box needs GPU only for the v0.2 KITTI-loader re-runs; D3 probe is
   a ~30-min task.
