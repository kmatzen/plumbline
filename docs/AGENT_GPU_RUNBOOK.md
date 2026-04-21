# Autonomous-agent GPU runbook

You are a Claude Code agent dropped onto a fresh GPU rental box. Your
mission is to **validate as many model × dataset × metric combinations
as possible against published paper numbers**, in priority order, then
report back. After validation lands, future sessions will fill in the
matrix.

This doc is the spec — everything you need is in here. The
human-oriented runbook at `GPU_RUNBOOK.md` covers the same setup steps
in friendly prose; this doc is denser and ordered for execution.

---

## What "validated" means

A reproduction `<name>` is **validated** when:

1. `reproductions/<name>.yaml` declares `paper_reference.value` (non-null)
   AND `paper_reference.source_confidence: verified_pdf`.
2. `plumbline reproduce <name> -o /tmp/results/<name>.json` exits 0.
3. The observed primary metric is within `tolerance_relative` of the
   paper value.

Anything else is **not validated** — record what happened and move on.
Do NOT modify YAMLs to make a number "fit." Do NOT invent paper values.
Do NOT downgrade `source_confidence` to make a failure pass.

The list of validation candidates is whatever
`scripts/list_validation_targets.py` emits (see § 4 — if that script
doesn't exist yet, derive it inline by grepping
`reproductions/*.yaml` for `source_confidence: verified_pdf` blocks).

---

## 1. Pre-flight (no GPU work yet)

Run these in order. **If any check fails, stop and report — do not
improvise around a failed prerequisite.**

```bash
# 1a. GPU visible?
nvidia-smi || { echo "ERROR: no GPU"; exit 1; }

# 1b. AWS session creds present?
aws sts get-caller-identity || {
  echo "ERROR: AWS session creds missing. User should run"
  echo "scripts/gpu_box_session_token.sh on their laptop and paste output here."
  exit 1
}

# 1c. HuggingFace login (rate-limited downloads otherwise). The CLI
# is now called `hf`; `huggingface-cli` is deprecated in hf-hub >=1.10.
hf auth whoami 2>/dev/null || {
  if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
  else
    echo "WARN: no HF login. DA-V2 / DA3 / Metric3D may rate-limit."
  fi
}

# 1d. Workaround for an HF xet-downloader multiprocess cleanup crash
# (hit during laptop-side weight staging on 2026-04-20). Always set
# this env var for the whole agent session — the non-xet path is
# just as fast on modern connections.
export HF_HUB_DISABLE_XET=1

# 1e. Disk: need ≥ 80 GB free for full Tier-1 sweep.
df -BG --output=avail / | tail -1
# If < 80 GB available, skip phases that need extra data (DTU, ETH3D).
```

---

## 2. Box setup

```bash
# Clone the repo.
git clone https://github.com/kmatzen/plumbline.git ~/plumbline
cd ~/plumbline

# Install. The "models" extra brings in torch+transformers+diffusers etc.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra models

# Verify torch sees the GPU.
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# Verify the CLI works.
uv run plumbline list-models
```

Set the dataset-root env vars in your shell rc so they persist across
re-shells:

```bash
export NYUV2_ROOT=$HOME/data/nyuv2
export KITTI_ROOT=$HOME/data/kitti
export DIODE_ROOT=$HOME/data/diode
export ETH3D_ROOT=$HOME/data/eth3d
export DTU_ROOT=$HOME/data/dtu
export GSO_ROOT=$HOME/data/gso
export SEVEN_SCENES_ROOT=$HOME/data/7scenes
export IBIMS1_ROOT=$HOME/data/ibims1
mkdir -p $NYUV2_ROOT $KITTI_ROOT $DIODE_ROOT $ETH3D_ROOT $DTU_ROOT \
         $GSO_ROOT $SEVEN_SCENES_ROOT $IBIMS1_ROOT
```

---

## 3. Data staging

**Strategy:** pull from the S3 cache first; only fetch from upstream
when the cache misses. Push back to S3 after every successful fetch
so the next session inherits the work.

**Preferred path — one command does it all:**

```bash
scripts/stage_all_data.sh
```

This wrapper syncs three things: `s3://plumbline-bench/datasets/` →
`~/data/`, `s3://plumbline-bench/hf-cache/` → `~/.cache/huggingface/`
(for HF-backed adapters), and `s3://plumbline-bench/torch-hub-cache/hub/`
→ `~/.cache/torch/hub/` (for Metric3D-v2, which uses `torch.hub.load`).
Then writes the dataset-root env vars to `~/.bashrc-plumbline`.
Source that file and you're ready to run.

**Manual path** (if `stage_all_data.sh` is unavailable for some reason):

```bash
aws s3 sync s3://plumbline-bench/datasets/ ~/data/ \
    --exclude '*/.plumbline_manifest/*' \
    --exclude '*/__pycache__/*'
aws s3 sync s3://plumbline-bench/hf-cache/ ~/.cache/huggingface/
aws s3 sync s3://plumbline-bench/torch-hub-cache/hub/ ~/.cache/torch/hub/
```

For each dataset, check what's present and fetch what's missing. The
**order below is the order you should stage in** — earliest items
unblock the most reproductions per GB.

| Order | Dataset | If missing, run |
|---|---|---|
| 1 | NYUv2 (3 GB) | `curl -L -O https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat` into `$NYUV2_ROOT` |
| 2 | KITTI raw + annotated depth (~6 GB) | `scripts/fetch_kitti.py --kitti-root $KITTI_ROOT` then unzip annotated-depth (see GPU_RUNBOOK § KITTI) |
| 3 | iBims-1 (40 MB) | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'iBims-1.zip' --local-dir /tmp/moge && unzip /tmp/moge/iBims-1.zip -d $IBIMS1_ROOT/..` |
| 4 | GSO (2 GB) | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'GSO.zip' --local-dir /tmp/moge && unzip /tmp/moge/GSO.zip -d $GSO_ROOT/..` |
| 5 | DIODE val (2.8 GB) | `curl -L -O http://diode-dataset.s3.amazonaws.com/val.tar.gz && tar xzf val.tar.gz -C $DIODE_ROOT/..` |
| 6 | ETH3D 3-scene subset (8 GB) | See GPU_RUNBOOK § ETH3D for the per-scene loop (3 scenes: courtyard, delivery_area, facade) |
| 7 | DTU test + Points (7.5 GB across two archives) | See GPU_RUNBOOK § DTU — `gdown` for the 554 MB MVSNet test zip + `aria2c` for the 6.97 GB Points.zip. Do NOT fetch `SampleSet.zip` — that's a 2-scan demo, NOT the eval set. |
| 8 | 7-Scenes (12 GB) | See GPU_RUNBOOK § 7-Scenes |

After each successful fetch:

```bash
aws s3 sync $<DATASET>_ROOT/ s3://plumbline-bench/datasets/<dataset>/ \
    --exclude '*/.plumbline_manifest/*'
```

**Skip a dataset if its fetch fails or the disk is tight.** Move on;
its dependent reproductions will be flagged "skipped — data missing"
in the report.

### 3.1 Model weights — lessons-learned gotchas

Most weights come down for free via the `hf-cache/` S3 sync. If a
model is NOT in the cache and you need to fetch it from HF Hub,
watch out for these (all hit during the initial cache population):

**(a) Many HF repos ship both `model.pt` AND `model.safetensors`.**
A plain `hf download <repo>` pulls both — doubling disk usage. For
repos that just store a single checkpoint in two formats (VGGT-1B,
some older DUSt3R-era repos), use a selective fetch:

```bash
hf download facebook/VGGT-1B \
    --include 'model.safetensors' --include '*.json' --include 'README.md'
```

`PyTorchModelHubMixin` (used by VGGT and similar) prefers
`model.safetensors` over `pytorch_model.bin` — `model.pt` won't
be loaded even if present.

**(b) `hf download` with HF's xet parallel-chunk downloader can
crash in a multiprocessing cleanup edge case** (Python 3.12 +
hf-hub 1.11, resource-tracker `__del__` failure mid-transfer).
The pre-flight in § 1 sets `HF_HUB_DISABLE_XET=1` globally which
forces the classical hf-hub downloader path. Keep it set.

**(c) Diffusers-style repos (Marigold, GeoWizard, etc.) split
weights across subfolders** — `unet/`, `vae/`, `text_encoder/`,
`scheduler/` — each with its own safetensors. `--include '*.safetensors' '*.json'`
matches all of them. Don't exclude the text_encoder even if you
don't think you need it — some pipeline classes fail to
instantiate without it.

**(d) After an interrupted `hf download`, delete leftover
`.incomplete` blobs before retrying:**

```bash
find ~/.cache/huggingface/hub -name '*.incomplete' -delete
```

---

## 4. Enumerate validation targets

Build the priority queue:

```bash
scripts/list_validation_targets.py --format json > /tmp/queue.json
# Or: scripts/list_validation_targets.py --format md for a human-readable preview.
```

The script already sorts the queue cheapest-first (runtime bucket
ascending, then name). It only returns rows with
`source_confidence: verified_pdf` and a non-null `value`.

Rough cost ranks embedded in the script (RTX 3090 / 4090):

| Bucket | Wall time per repro | Examples |
|---|---|---|
| **fast** (~1-3 min) | DA-V2 S/B/L NYU + KITTI, MoGe-1 NYU, DA3 NYU |
| **medium** (~5-10 min) | Metric3D-v2 L NYU+KITTI, Marigold NYU, Depth Pro KITTI, all DIODE |
| **slow** (~30-60 min) | Metric3D-v2 Giant variants, ETH3D multi-scene, DTU full-22 |

Run **fast** first. If any fast row fails outright (not "off paper" —
*fails*), stop and report — something fundamental is broken.

---

## 5. Run the matrix

```bash
mkdir -p /tmp/results
mkdir -p /tmp/logs
```

For each `<name>` in the priority queue:

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
uv run plumbline reproduce "$name" -o "/tmp/results/${name}.json" \
    > "/tmp/logs/${name}.log" 2>&1
ec=$?

# Log the outcome.
python3 -c "
import json, os, sys
name = '$name'; ec = $ec
result = {'name': name, 'exit_code': ec, 'ts': '$ts'}
try:
    with open(f'/tmp/results/{name}.json') as f:
        r = json.load(f)
    result['observed'] = r.get('aggregate_metrics', {}).get(r.get('primary_metric', ''))
    result['paper_match'] = r.get('paper_match')
except Exception as e:
    result['error'] = str(e)
print(json.dumps(result))
" >> /tmp/results/index.jsonl

# Push results back as we go (so a crash mid-session doesn't lose work).
aws s3 sync /tmp/results/ "s3://plumbline-bench/runs/${ts}/results/"
aws s3 sync /tmp/logs/ "s3://plumbline-bench/runs/${ts}/logs/"
```

After each run, also push the prediction cache so the next session
can re-run with different alignments / metrics without re-inferring:

```bash
aws s3 sync ~/.cache/plumbline/predictions/ \
    s3://plumbline-bench/predictions/
```

### Failure handling

| Symptom | What to do |
|---|---|
| `DatasetNotAvailable` | Mark "skipped — data missing"; continue |
| `torch.cuda.OutOfMemoryError` | Retry with `--max-views` halved (multi-view models); if still fails, mark "skipped — OOM" |
| Other exception | Capture full traceback to log; mark "failed"; continue |
| Exit 0 + paper_match true | "✅ MATCH" |
| Exit 0 + paper_match false | "⚠️ off paper" (record observed vs published) |
| Exit 0 + paper_match null (no paper target) | "ℹ️ informational" — should not happen for verified_pdf rows; flag for human review |

**Never:** modify YAMLs, `git commit`, push to GitHub, edit
`paper_reference.value`, edit tolerances, or "fix" a failure by
relaxing the protocol. If a reproduction fails, the answer is in the
report — not in the YAML.

---

## 6. Smoke-test new adapters / loaders without paper rows

These shipped as adapters/loaders but have no `verified_pdf`
reproduction yet. Validate that they at least **run end-to-end** on
real data without crashing:

```bash
# GeoWizard on NYU (paper protocol unverified; just smoke):
uv run plumbline run --model geowizard --dataset nyuv2 \
    --tasks mono_depth --scale-alignment scale_shift_depth \
    --max-views 1 -o /tmp/results/_smoke_geowizard_nyuv2.json

# π³ on ETH3D 3-scene (multi-view smoke):
uv run plumbline run --model pi3 --dataset eth3d \
    --tasks mvs_depth pose --max-views 8 \
    -o /tmp/results/_smoke_pi3_eth3d.json

# 7-Scenes loader smoke (any model that does pose):
uv run plumbline run --model mast3r --dataset 7scenes \
    --tasks pose --max-views 2 \
    -o /tmp/results/_smoke_mast3r_7scenes.json

# iBims-1 loader smoke (any mono-depth model):
uv run plumbline run --model depth-anything-v2 --dataset ibims1 \
    --tasks mono_depth --scale-alignment scale_shift \
    --max-views 1 -o /tmp/results/_smoke_dav2_ibims1.json
```

For these, "validated" means: exits 0, metric values are finite, no
out-of-distribution shapes (e.g. depth in [0, 1] when expecting
meters). Note the observed value in the report so the human can later
pin a reference target.

---

## 7. Final report

When the queue is exhausted (or the budget is hit), generate the
report:

```bash
uv run python <<'PY' > /tmp/results/REPORT.md
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("/tmp/results/index.jsonl").read_text().splitlines() if l.strip()]
print("# GPU validation report\n")
print("| Reproduction | Status | Observed | Notes |")
print("|---|---|---|---|")
for r in rows:
    name = r['name']
    if r.get('paper_match') is True:    status = "✅ MATCH"
    elif r.get('paper_match') is False: status = "⚠️ OFF PAPER"
    elif r.get('exit_code') != 0:       status = "❌ FAILED"
    else:                                status = "ℹ️ INFO"
    obs = r.get('observed', '—')
    notes = r.get('error', '')[:80]
    print(f"| `{name}` | {status} | {obs} | {notes} |")
PY

aws s3 cp /tmp/results/REPORT.md s3://plumbline-bench/runs/${ts}/REPORT.md
cat /tmp/results/REPORT.md
```

Print the report to stdout so it's visible in the agent's transcript,
AND push to S3 so the user can pull it from the laptop:

```bash
aws s3 cp s3://plumbline-bench/runs/${ts}/REPORT.md ~/Downloads/
```

---

## 8. Tear down

When done, **do not** delete the S3 cache (that's the whole point of
caching across rentals). Do **not** terminate the rental box yourself
— leave it for the user to reclaim.

Stop work and report when:
- Every `verified_pdf` reproduction has been attempted (success or
  documented failure).
- The user-set time / cost budget is hit (default: 4 hours wall time,
  ~$3-5 on a 4090 at $0.30/hr).
- Disk is exhausted (run `df -BG /` periodically; halt new fetches at
  90 % full).
- A pre-flight assumption broke mid-run (e.g. AWS session expired —
  ask the user to mint a new token; do NOT use other credentials).

---

## 9. What you must NEVER do

These rules are **not** "best practices" — they are hard constraints.
A run that violates them is worse than a run that bails early.

1. **Never modify reproduction YAMLs.** Not the paper value, not the
   tolerance, not the protocol, not the citation, not the
   `source_confidence`. The YAMLs are the agreement with the paper;
   you don't get to renegotiate.
2. **Never invent paper numbers.** If a paper value seems wrong, raise
   it in the report — don't "correct" it.
3. **Never `git commit`, `git push`, or `gh pr create`.** Code changes
   come from the laptop, not from this box.
4. **Never delete the S3 cache** or anything in `s3://plumbline-bench/`
   that pre-dates this session.
5. **Never use credentials other than the session token the user
   provided.** No copying long-lived keys onto the box; no
   instance-profile fallbacks.
6. **Never bypass `pre-commit` / lint / test failures with `--no-verify`
   or similar.** You are not committing in the first place; if a hook
   would block you, that's evidence something is wrong upstream that
   needs the user's attention.
7. **Never re-download a dataset that's already on disk.** Always
   check first; the prediction cache + dataset cache are precious.

If you find yourself wanting to do any of these — stop, write what you
were about to do into the report, and end the session for the user to
review.

---

## 10. Communication back to the user

This box has no chat connection back to the user. Two channels:

- **stdout** — your transcript is captured by Claude Code's session
  log. Print enough that a human reading the transcript later can
  follow what you did and why, especially decisions to skip / retry /
  bail.
- **S3** — the report at `s3://plumbline-bench/runs/<ts>/REPORT.md` is
  the durable artifact. The user pulls it after the session.

End your final transcript message with:
1. The full markdown report (also pushed to S3).
2. A one-paragraph summary of what worked, what didn't, and what the
   highest-leverage next step is.
3. The S3 paths to the report, results JSON, and prediction cache.

Then stop. Don't loop.
