---
name: reproduce-pose
description: Stage a pose dataset (RealEstate10K, CO3Dv2, or the Sintel / TUM-Dynamics trajectory sets) and run a pose reproduction on the anima GPU box, then compare to the paper target — pairwise mAA@30 (mast3r/dust3r/vggt) or trajectory ATE/RPE (dage/cut3r/monst3r, DAGE/MonST3R Table 4). Use when asked to reproduce, re-run, or validate a pose cell, or to stage RealEstate10K / CO3Dv2 / TUM-Dynamics pose data.
---

# Reproduce a pose cell

End-to-end recipe for the `*-pose` reproduction cells. Two families:
- **Pairwise** (`pairwise_pose_auc@30`, a.k.a. mAA@30) — CO3Dv2 / RealEstate10K,
  10-frame clips, all-pairs relative pose. Models: mast3r / dust3r / vggt.
- **Trajectory** (`trajectory_ate_rmse` + RPE, DAGE/MonST3R Table 4) — Sintel /
  TUM-Dynamics, one full-clip Sample/scene, Sim(3)-aligned ATE/RPE via `evo`.
  Models: dage / cut3r / monst3r.

Runs on **anima** (`ssh anima-claude`, GTX 1080 Ti, 11 GB). See `GPU_RUNBOOK.md`
for the general runbook and `REPRODUCTIONS.md` for cell targets.

## 0. Pick the cell and its data

**Pairwise (mAA@30):**

| cell | dataset | paper mAA@30 | stage with |
|---|---|---|---|
| `mast3r-co3dv2-pose` / `dust3r-co3dv2-pose` / `vggt-co3dv2-pose` | CO3Dv2 | 0.818 / 0.772 / 0.882 | `scripts/co3dv2_prefetch.py` |
| `mast3r-realestate10k-pose` / `dust3r-…` / `vggt-…` | RealEstate10K | 0.764 / 0.612 / 0.853 | `scripts/stage_realestate10k.py` |

**Trajectory (ATE, DAGE/MonST3R Table 4):**

| cell | dataset | paper ATE | stage with |
|---|---|---|---|
| `dage-sintel-pose` / `cut3r-sintel-pose-dage` / `monst3r-sintel-pose` | Sintel | 0.132 / 0.217 / 0.108 | MPI-Sintel (`$SINTEL_ROOT`) |
| `dage-tum-pose` ✅ / `cut3r-tum-pose-dage` | TUM-Dynamics | 0.014 / 0.047 | `scripts/stage_tum_dynamics.py` |
| `pi3-sintel-pose-dage` (Ampere+ only) | Sintel | 0.074 | `plumbline install pi3` |

dage (fp16) + cut3r (online) fit the 1080 Ti on these clips; `dage-tum-pose`
landed ✅ **0.0136 vs 0.014** (8/8, 2026-06-03). pi3 / VGGT trajectory cells are
compute-blocked on the 1080 Ti (Pascal fp16 too slow / bf16; `max_views` cap) —
need an Ampere+ card. Set `max_views` ≥ clip length (TUM clips are exactly 90;
the trajectory metric needs pred and GT the same length).

VGGT is **bf16** → won't perform on the Pascal 1080 Ti; it needs an H100-class
card. mast3r/dust3r run on anima (slow PyTorch RoPE fallback — correct, just
~60 s/clip dust3r, ~2–3 min/clip mast3r `sparse_ga`).

## 1. Check disk FIRST

anima runs at ~98 % disk. **Always** `ssh anima-claude 'df -h /'` before staging.
CO3Dv2 metadata is ~70 MB/category; RealEstate10K frames are tiny but transient
videos are not. If `< ~2 GB` free, stage a smaller subset (fewer CO3Dv2
categories; fewer RealEstate10K clips) and `df -h /` again mid-stage.

## 2. Stage the data

**RealEstate10K** (frames scraped from YouTube, ~94 % hit rate):
```bash
# one-time: test camera files (stream-extract test/ only; storage.googleapis.com is anon-public)
ssh anima-claude 'mkdir -p ~/data/re10k_meta && cd ~/data/re10k_meta && \
  curl -sL https://storage.googleapis.com/realestate10k-public-files/RealEstate10K.tar.gz | tar -xzf - RealEstate10K/test/'
# scrape N usable clips (resumable; ~480p, deletes each video after extracting frames)
ssh anima-claude 'cd ~/git/geobench && .venv/bin/python scripts/stage_realestate10k.py \
  --meta ~/data/re10k_meta/RealEstate10K/test --out ~/data/realestate10k --target 50'
```

**CO3Dv2** (selective HTTP-Range over per-category zips):
```bash
ssh anima-claude 'cd ~/git/geobench && CO3DV2_ROOT=~/data/co3dv2 .venv/bin/python \
  scripts/co3dv2_prefetch.py --root ~/data/co3dv2 --sequences-per-category 5 \
  --categories apple backpack banana bench bottle hydrant --seed 0'
```
For a bounded subset, pass matching `categories` + `sequences_per_category` to
BOTH the prefetch and the eval (a temp YAML copy with
`dataset.kwargs.{categories,sequences_per_category}`) so the staged clips are
exactly what the loader's seeded sampler requests.

**TUM-Dynamics** (public freiburg3, no ToS; member-selective — only the 90 eval
frames/seq, ~366 MB total):
```bash
ssh anima-claude 'cd ~/git/geobench && .venv/bin/python scripts/stage_tum_dynamics.py \
  --out ~/data/tum_dynamics'   # then $TUM_ROOT=~/data/tum_dynamics
```
**Sintel** is already at `~/data/sintel` (`$SINTEL_ROOT`); the loader's `full_seq`
mode emits one trajectory Sample/scene over the 14 dynamic-final clips.
**Manifest gotcha:** the TUM loader keys its manifest cache on the set
of *present* sequences, so staging more data after a partial scan invalidates it
— but if you ever see fewer samples than expected, `rm -rf
~/data/<set>/.plumbline_manifest` and re-run.

## 3. Run the reproduction (detached — anima SSH drops)

Launch with `setsid nohup … < /dev/null &` (NOT a heredoc — that ties stdin to
the flaky SSH and dies on drop). One unique log path per launch.
```bash
ssh anima-claude 'cd ~/git/geobench && setsid nohup env \
  REALESTATE10K_ROOT=$HOME/data/realestate10k DUST3R_ROOT=$HOME/deps/dust3r \
  .venv/bin/plumbline reproduce mast3r-realestate10k-pose -o /tmp/out.json \
  > /tmp/run.log 2>&1 < /dev/null & echo pid $!'
```
- mast3r uses the vendored `_vendor/mast3r` (no `MAST3R_ROOT` needed); its pose
  defaults to `pose_backend="sparse_ga"` (matching-based — the faithful path).
  Pass `model.kwargs.pose_backend=dust3r_ga` to A/B against the legacy path.
- dust3r needs `DUST3R_ROOT=~/deps/dust3r`. Checkpoints auto-download from HF.

Wait for it with a poll loop (don't hold the SSH): poll for the output JSON or
`! kill -0 <pid>`. Watch the GPU with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`; a
killed parent shell can orphan a python child still holding the card — kill the
**python** pid, not just the shell.

## 4. Read the result

```bash
ssh anima-claude 'cd ~/git/geobench && .venv/bin/python -c "import json; \
  m=json.load(open(\"/tmp/out.json\"))[\"aggregate_metrics\"]; \
  print(\"mAA@30\", m[\"pairwise_pose_auc@30\"], \"RTA@5\", m[\"pairwise_RTA@5\"])"'
```
Compare to the paper target (±5 %). On a small/easy subset both dust3r and
mast3r overshoot the paper by ~8–11 % (subset effect) — judge the
**dust3r↔mast3r gap**, not the absolute: matching (`sparse_ga`) should put
MASt3R well above DUSt3R on wide-baseline RE10K (~+15 pt) but only marginally
above on narrow-baseline CO3Dv2 (~+2 pt). A near-tie on RE10K means the
matching path isn't engaged.

For **trajectory** cells, read the ATE/RPE instead (the `plumbline reproduce`
markdown prints them, or from JSON):
```bash
ssh anima-claude 'cd ~/git/geobench && .venv/bin/python -c "import json; \
  m=json.load(open(\"/tmp/out.json\"))[\"aggregate_metrics\"]; \
  print(\"ATE\", m[\"trajectory_ate_rmse\"], \"RPE-t\", m[\"trajectory_rpe_trans_rmse\"], \
  \"RPE-r\", m[\"trajectory_rpe_rot_deg_rmse\"])"'
```
Tolerance is ±10 % (`pose_trajectory_metrics`). TUM ATEs are sub-cm, so the band
is tight — read companions + the cut3r baseline before calling a miss. ATE is
Sim(3)-global (subsampling-robust); RPE is consecutive-frame (inflates if you
stride the clip). Confirm `Samples evaluated: N/N` matches the full clip count.

## 5. Clean up

Remove staged subsets + `/tmp/*.json,*.log` to reclaim the tight disk
(`~/data/co3dv2` deletion needs explicit user OK — shared box).
