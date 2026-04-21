# Session handoff — laptop prep complete

Last updated: 2026-04-21. This doc is the short "pick up here" note
for the next GPU-rental session (human or Claude-Code agent).

## Status

**Laptop-side Phase-A prep is complete.** The next action is booking a
GPU rental box and running `scripts/stage_all_data.sh` followed by
`plumbline reproduce` on each target in the validation queue.

## Validation queue

Run `scripts/list_validation_targets.py --format md` for the current
list (emits in priority order: fast → medium → slow buckets).

At 2026-04-21 the queue has **21 `source_confidence: verified_pdf`
paper-match targets**:

- **8 fast** (~1–3 min each): DA-V2 Small/Base/Large on NYU+KITTI,
  DA3 on NYU, MoGe-1 on NYU.
- **9 medium** (~5–10 min each): GeoWizard NYU+KITTI, Marigold NYU+
  KITTI, Metric3D-v2 L on NYU+KITTI, MoGe-1 on DIODE (indoor+both)
  and KITTI.
- **4 slow** (~30–60 min each): Metric3D-v2 Giant on NYU+KITTI, VGGT
  ETH3D multiscene (Table 3), VGGT DTU MVS (Table 2, the v0.1 gate).

Plus 2 informational smoke YAMLs (`pi3_dtu_mvs`, `pi3_eth3d_multiscene`)
that run π³ and report observed numbers for a future unit-convention
check against its paper.

Rough total: ~4 h wall on a 3090/4090 to clear the full paper-match
queue, $2–$4 at current rental prices.

## What's pre-staged in S3

`s3://plumbline-bench/` (us-west-2), 7,287 objects / 54 GB:

```
datasets/
  nyuv2/                2.97 GB   full Silberman/Eigen labeled .mat
  kitti/                0.54 GB   652 Eigen-benchmark frames
                                  (pruned from 8.5 GB raw) + calib
  dtu/                  4.90 GB   22 MVSNet test scans + GT Points
                                  (NOT SampleSet.zip — see note in
                                  REPRODUCTIONS.md § DTU protocol)
  eth3d/                3.37 GB   3-scene subset: courtyard +
                                  delivery_area + facade (undistorted
                                  + scan_eval)
  ibims1/               0.04 GB   MoGe-preprocessed bundle
  gso/                  0.09 GB   MoGe-preprocessed bundle
hf-cache/              35.00 GB   7 HF repos: DA-V2-{Small,Base,Large}-hf,
                                  DA3-LARGE-1.1, moge-vitl, VGGT-1B,
                                  Marigold-v1-1. All safetensors-only
                                  (not model.pt duplicates).
torch-hub-cache/        7.37 GB   Metric3D-v2 ViT-{Small,Large,Giant2}
                                  weights + YvanYin_Metric3D repo tree.
```

Monthly storage cost: ~$1.30.

## How the agent gets onto a box

One-time laptop setup (only needs to be done once, ever):

```bash
aws configure --profile plumbline-gpu-cache
# paste the long-lived IAM user keys (provisioned 2026-04-20)
```

Per-rental:

```bash
# On your laptop — mints a 12 h session token.
scripts/gpu_box_session_token.sh
# Paste the exported env vars onto the rental box.
```

On the rental box:

```bash
# Install tools.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone git@github.com:kmatzen/plumbline.git
cd plumbline
uv sync --extra models --extra scripts

# Stage all cached artifacts.
scripts/stage_all_data.sh
source ~/.bashrc-plumbline

# Upstream clones for the three adapters that don't ship on PyPI:
git clone https://github.com/naver/mast3r        --recursive /workspace/deps/mast3r
git clone https://github.com/fuxiao0719/GeoWizard /workspace/deps/geowizard
git clone https://github.com/yyfz/Pi3             /workspace/deps/pi3
export MAST3R_ROOT=/workspace/deps/mast3r
export GEOWIZARD_ROOT=/workspace/deps/geowizard
export PI3_ROOT=/workspace/deps/pi3

# Metric3D-v2's code deps (not on the models extra):
uv pip install mmengine mmcv-lite

# Validation loop.
scripts/list_validation_targets.py --format json > /tmp/queue.json
# (Run each per docs/AGENT_GPU_RUNBOOK.md § 5)
```

Full human-oriented walkthrough: `GPU_RUNBOOK.md`.
Full agent playbook (denser, ordered for execution): `docs/AGENT_GPU_RUNBOOK.md`.

## Hard constraints the agent must respect

1. **Never modify reproduction YAMLs.** If a paper number doesn't
   reproduce, that's a finding, not a parameter to tune.
2. **Never commit, push, or open PRs.** Code changes come from the
   laptop.
3. **Never invent paper numbers.** `source_confidence: verified_pdf`
   is a contract with the paper's PDF; if a cell isn't in the paper,
   leave `value: null`.
4. **Never delete S3 cache contents.** They're shared across future
   sessions.
5. **Never use credentials other than the session token.** No
   copying long-lived keys onto the rental box.
6. **Never bypass hooks with `--no-verify`** or similar.

The 2026-04-20 audit codifying these rules lives in
`reproductions/AUDIT.md`; the underlying feedback memory is
`feedback_paper_citations.md` in the agent's long-term store.

## Deferred to v0.2

Documented in `plan.md § 10` but not on the v0.1 critical path:

- Depth Pro paper rows (paper evaluates Sun-RGBD / ETH3D /
  Middlebury / etc. — none currently loaded except ETH3D; would
  need new loaders).
- Pi3 verified-PDF pinning (multi-view chamfer units unclear until
  first-run observation).
- 7-Scenes + Co3Dv2 pose benchmarks (loaders wired, data not staged).
- MoGe-2 / MASt3R / Pi3 HF weight staging (no verified-PDF
  reproductions currently target those repos).
- DIODE outdoor protocol (combined val is 6× off paper; suspected
  wrong depth_clip + missing sky mask).
- Protocol preset for Sintel (auth-gated; deprioritized on
  2026-04-19 pivot).

## How to read the session trail

- `git log --oneline origin/main` — all commits. The last ~15 cover
  this prep session.
- `reproductions/AUDIT.md` — paper-number audit (24 YAMLs checked,
  9 originally clean, rest corrected/downgraded).
- `docs/SAMPLE_LISTS.md` — per-reproduction sample-selection
  inventory (IN-REPO / LOADER-DEFAULT / SCENE-FILTER tags).
- `docs/dataset_footprints.md` — per-dataset disk footprint table
  + Tier-1 session budget example (~60 GB).
- `docs/AGENT_GPU_RUNBOOK.md` — the agent's playbook.
- `GPU_RUNBOOK.md` — the human's walkthrough.
- `REPRODUCTIONS.md` — per-YAML status matrix.
