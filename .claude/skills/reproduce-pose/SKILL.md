---
name: reproduce-pose
description: Stage a multi-view pose dataset (RealEstate10K, CO3Dv2) and run a mast3r/dust3r/vggt pose reproduction on the anima GPU box, then compare mAA@30 to the paper target. Use when asked to reproduce, re-run, or validate a pose cell, or to stage RealEstate10K / CO3Dv2 pose data.
---

# Reproduce a multi-view pose cell

End-to-end recipe for the `*-pose` reproduction cells (CO3Dv2 / RealEstate10K
`pairwise_pose_auc@30`, a.k.a. mAA@30). Runs on **anima** (`ssh anima-claude`,
GTX 1080 Ti, 11 GB). See `GPU_RUNBOOK.md` for the general runbook and
`REPRODUCTIONS.md` for cell targets.

## 0. Pick the cell and its data

| cell | dataset | paper mAA@30 | stage with |
|---|---|---|---|
| `mast3r-co3dv2-pose` / `dust3r-co3dv2-pose` / `vggt-co3dv2-pose` | CO3Dv2 | 0.818 / 0.772 / 0.882 | `scripts/co3dv2_prefetch.py` |
| `mast3r-realestate10k-pose` / `dust3r-…` / `vggt-…` | RealEstate10K | 0.764 / 0.612 / 0.853 | `scripts/stage_realestate10k.py` |

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

## 5. Clean up

Remove staged subsets + `/tmp/*.json,*.log` to reclaim the tight disk
(`~/data/co3dv2` deletion needs explicit user OK — shared box).
