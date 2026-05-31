# GPU runbook

Spin up → bring up → run → tear down. Single doc; the agent and the
human see the same instructions.

## The run queue

The backlog of reproductions awaiting a GPU is a machine-readable file,
`reproductions/gpu_queue.yaml`, driven by the `plumbline queue` command —
this is the executable index for everything below.

```bash
plumbline queue                  # plan: pending jobs, footprints, env vars
plumbline queue --include-blocked  # also show what NOT to run (and why)
plumbline queue --run            # execute pending jobs in priority order
plumbline queue --run --name vggt-co3dv2-pose   # just one
plumbline queue --run -o /tmp/results/queue.json  # write a JSON summary
```

`--run` executes each `pending` job via `plumbline reproduce`, captures
MATCH / MISMATCH / INFO / ERROR, and keeps going if one job fails. It
**never** runs `blocked` jobs (upstream-blocked cells, data-not-staged),
and never mutates a YAML. After a job lands within tolerance, hand-flip
its `status:` to `done` in the queue file and update `REPRODUCTIONS.md`.

Use the listing to size the box: sum the `GB` column for the jobs you
intend to run (+ weights + cache + 20 % headroom) against the rental
disk before booking. As of 2026-05-30 the CO3Dv2 pose cells and the
mono-depth coverage backlog (GSO / iBims-1 / DIODE-moge / KITTI-moge /
ETH3D-moge / iBims-1, all PDF-verified) are **done**. Remaining queue work is
mostly **blocked** off-paper investigations (native ETH3D/Sintel Table 2 →
handoff docs; DIODE native D29 outdoor) plus **`depth-pro-sintel`** (Depth Pro
Table 1 δ₁ experiment — upstream ships no Sintel eval script). Use
`plumbline queue --include-blocked` before re-running anything. Active execution
plan: [`docs/GPU_BACKLOG_PLAN.md`](docs/GPU_BACKLOG_PLAN.md) (D29 MoGe warp on
native DIODE is track 1).

## Hard constraints

1. **Never modify reproduction YAMLs.** Failed paper-match is a finding,
   not a parameter to tune.
2. **Never invent paper numbers.** Suspected wrong → flag in the report.
3. **Never delete the S3 cache.** Shared across sessions.
4. **Never use credentials other than the session token the user
   provides.** Token expires in ≤12 h.
5. **Never bypass `pre-commit` / lint / test failures with `--no-verify`.**
6. **Never re-download a dataset that's already on disk.**
7. **Never bulk-pull `s3://plumbline-bench/datasets/`.** Pull only what
   the current sample / reproduction needs (see § Thrift bootstrap).

## Pod disk policy (k8s / vast.ai-style boxes)

The **home overlay is small** (~75G). HuggingFace/uv/torch caches,
MoGe-eval zip downloads, and dataset trees belong on **`/mnt/localssd`**
— not `$HOME/data` or `$HOME/.cache`. On this cluster, using more than
**1/8 of the node's local SSD** can get the pod killed (~3 TiB budget on
a 26 TiB volume; check with `df -h /mnt/localssd`).

```bash
source /mnt/localssd/plumbline/scripts/pod-localssd-env.sh
# → PLUMBLINE_WORK=/mnt/localssd/plumbline-work
# → HF_HOME, UV_CACHE_DIR, DDAD_MOGE_ROOT, SINTEL_MOGE_ROOT, DAV2_ROOT, …

df -h / /mnt/localssd
du -sh "$PLUMBLINE_WORK"   # keep an eye on the 1/8 budget during big fetches
```

Stage downloads under `"$PLUMBLINE_WORK/data/..."`, write result JSONs to
`"$PLUMBLINE_WORK/runs/"`, and clone adapter deps to
`"$PLUMBLINE_WORK/deps/"`. Do **not** re-download into `/tmp` on the overlay
and leave the zips behind.

**Backup habit (ephemeral pods).** After each successful reproduction (or
before stepping away), push artifacts off the box:

```bash
source scripts/pod-localssd-env.sh
./scripts/backup-session.sh <session-tag>   # runs/*.json + MoGe bundles → S3
git push origin main                         # code + queue bookkeeping
```

Session token exports expire (~12 h); re-paste when `aws s3 ls` fails.

## Pre-flight

```bash
nvidia-smi || { echo "no GPU"; exit 1; }
aws sts get-caller-identity || {
  echo "Mint a session token: scripts/gpu_box_session_token.sh on laptop, paste here."
  exit 1
}
hf auth whoami 2>/dev/null || hf auth login --token "$HF_TOKEN" --add-to-git-credential
export HF_HUB_DISABLE_XET=1   # workaround for xet-downloader multiprocess crash
df -BG --output=avail / | tail -1   # need ≥ 30 GB for a typical session
command -v s5cmd || {
    curl -sSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
      -o /tmp/s5cmd.tgz && tar -xzf /tmp/s5cmd.tgz -C /tmp s5cmd && \
      mv /tmp/s5cmd $HOME/.local/bin/
}
```

## Box setup

```bash
git clone https://github.com/kmatzen/plumbline.git ~/plumbline
cd ~/plumbline
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python -c "import torch; assert torch.cuda.is_available()"
uv run plumbline list-models
```

After `source scripts/pod-localssd-env.sh`, add any **non-MoGe-bundle**
roots you need. Don't pre-create directories for datasets you won't touch.

```bash
export NYUV2_ROOT=$PLUMBLINE_WORK/data/nyuv2
export KITTI_ROOT=$PLUMBLINE_WORK/data/kitti
export ETH3D_ROOT=$PLUMBLINE_WORK/data/eth3d
export DTU_ROOT=$PLUMBLINE_WORK/data/dtu
# MoGe-bundle roots (DDAD_MOGE_ROOT, SINTEL_MOGE_ROOT, …) are set by pod-localssd-env.sh
```

### Per-adapter extras (install only what you need)

The base install (`uv sync` / `pip install plumbline-bench`) covers
torch + diffusers + transformers — enough for DA-V2, DA3, Marigold,
Depth Pro, Metric3Dv2.

Don't hand-maintain install commands: ask the tool.
`plumbline install --list` prints the table below, `plumbline install
<adapter>` prints the full plan (add `--yes` to run it), and `plumbline
doctor` reports which adapters are present on the current box.

<!-- generated from src/plumbline/install.py INSTALL_SPECS; run `plumbline install --list` -->

| Adapter | Kind | Install plan |
|---|---|---|
| metric3d-v2 | base | (none — base install) |
| marigold | base | (none — base install) |
| depth-anything-v2 | base | (none — base install) |
| depth-anything-3 | pypi | `uv pip install depth-anything-3` |
| moge | git | `uv pip install 'git+https://github.com/microsoft/MoGe.git'` |
| vggt | git | `uv pip install 'git+https://github.com/facebookresearch/vggt'` |
| depth-pro | git | `uv pip install 'git+https://github.com/apple/ml-depth-pro.git'` |
| mast3r | clone | `git clone --recursive https://github.com/naver/mast3r $HOME/deps/mast3r; uv pip install roma scikit-learn trimesh; export MAST3R_ROOT=$HOME/deps/mast3r; export DUST3R_ROOT=...` (see notes) |
| dust3r | clone | `git clone --recursive https://github.com/naver/dust3r $HOME/deps/dust3r; uv pip install roma scikit-learn trimesh; export DUST3R_ROOT=$HOME/deps/dust3r` |
| monst3r | clone | `git clone --recursive https://github.com/Junyi42/monst3r $HOME/deps/monst3r; uv pip install roma scikit-learn trimesh; export MONST3R_ROOT=$HOME/deps/monst3r` |
| cut3r | clone | `git clone --recursive https://github.com/CUT3R/CUT3R $HOME/deps/cut3r; uv pip install -r $HOME/deps/cut3r/requirements.txt; export CUT3R_ROOT=$HOME/deps/cut3r; export CUT3R_CKPT=...` (512-DPT weights per repo README) |
| pi3 | clone | `git clone https://github.com/yyfz/Pi3 $HOME/deps/pi3; uv pip install -r $HOME/deps/pi3/requirements.txt; export PI3_ROOT=$HOME/deps/pi3` |
| geowizard | clone | `git clone https://github.com/fuxiao0719/GeoWizard $HOME/deps/geowizard; export GEOWIZARD_ROOT=$HOME/deps/geowizard` then `uv sync --extra geowizard; uv pip install --force-reinstall 'nvidia-cudnn-cu12==9.1.0.70'` |

If `uv pip install` from a git URL dies with `curl 92 HTTP/2 stream
CANCEL`, retry — pause concurrent s5cmd jobs first to free bandwidth.

**Metric3Dv2 gotcha:** if `xformers` is installed with a wheel that
doesn't match torch/CUDA exactly, the bundled dinov2 backbone raises
`NotImplementedError` in `memory_efficient_attention_forward` at
forward time. Either install a matching xformers wheel, or
`uv pip uninstall xformers` (Metric3D falls back to a pure-PyTorch
attention path; this is what plumbline's CI does).

## Thrift bootstrap

**Do not bulk-pull the dataset cache.** Per `plan.md § 12`, work is
single-record-diff: pick one sample, pull its inputs + GT, run, diff
against reference code. Full-dataset runs are the *last* step.

For a single-record diff:

```bash
# Pull just the sample under investigation
s5cmd cp 's3://plumbline-bench/datasets/eth3d/courtyard/000123*' \
    "$ETH3D_ROOT/courtyard/"

# Pull just the model weights for the issue
s5cmd cp -recursive 's3://plumbline-bench/hf-cache/models--facebook--VGGT-1B/*' \
    ~/.cache/huggingface/hub/models--facebook--VGGT-1B/

# Run reference code in parallel for the same sample (clone fresh each session)
git clone --depth 1 https://github.com/CUT3R/CUT3R /tmp/cut3r
# ... run reference, save tensors, diff vs plumbline output
```

For a full-dataset run (only after single-record diff has matched
reference within tolerance), see § Per-dataset fetch below — fetch the
specific dataset and the specific model weights, nothing else.

### Per-dataset fetch — minimum-viable footprints

Footprint is *minimum viable* (sample-list-driven), not full release.
Sum the rows for your queue + 20 % headroom and confirm it fits the
rental box's disk before starting.

**Prefer S3 first.** Most of these are already cached at
`s3://plumbline-bench/datasets/` (as of 2026-05-30: `nyuv2 kitti kitti_moge
diode diode_moge gso ibims1 eth3d eth3d_moge sintel dtu cut3r_eval`; DDAD/Sintel
MoGe zips may need HF fetch — see rows below). Pull
with `aws s3 sync s3://plumbline-bench/datasets/<name>/ $<ROOT>/` — faster than
re-fetching from source, and avoids the HF/source quirks. The source recipes
below are the fallback when a dataset isn't on S3.

**ROOT-points-at-parent gotcha:** the MoGe-bundle loaders expect the env var
to point at the directory *containing* the bundle subdir, not the subdir
itself — `DIODE_MOGE_ROOT/DIODE/`, `KITTI_MOGE_ROOT/KITTI/`,
`ETH3D_MOGE_ROOT/ETH3D/` (default `$PLUMBLINE_WORK/data/eth3d_moge`, not `moge_eval/`),
`DDAD_MOGE_ROOT/DDAD/`, `SINTEL_MOGE_ROOT/Sintel/` (under `$PLUMBLINE_WORK/data/moge_eval/`).
MoGe upstream DA-V2 audit: `scripts/run-moge-upstream-dav2.sh`.
(GSO/iBims point directly at the scene-dir parent.)

| Dataset | Min viable | Fetch |
|---|---|---|
| NYUv2 | 3 GB | `curl -L -O https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat` to `$NYUV2_ROOT` |
| KITTI Eigen-652 | 6 GB | `scripts/fetch_kitti.py --kitti-root $KITTI_ROOT` (selective per-drive) + `data_depth_annotated.zip` (~14 GB unpacked, but loader only needs the 652 listed frames) |
| KITTI MoGe-eval bundle | 3 GB | `hf download Ruicheng/monocular-geometry-evaluation KITTI.zip --repo-type dataset --local-dir /tmp/moge_dl && unzip /tmp/moge_dl/KITTI.zip -d $KITTI_MOGE_ROOT` |
| iBims-1 | 40 MB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'iBims-1.zip' --local-dir /tmp/moge && unzip /tmp/moge/iBims-1.zip -d $IBIMS1_ROOT/..` |
| GSO | 2 GB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'GSO.zip' --local-dir /tmp/moge && unzip /tmp/moge/GSO.zip -d $GSO_ROOT/..` |
| DIODE val | 2.8 GB | `curl -L -O http://diode-dataset.s3.amazonaws.com/val.tar.gz && tar xzf val.tar.gz -C $DIODE_ROOT/..` — native Table-2: see D29 handoff. MoGe bundle: `s3://plumbline-bench/datasets/diode_moge/` → `$DIODE_MOGE_ROOT` (not `moge_eval/`). |
?| ETH3D 3-scene (chamfer) | 8 GB | Per scene: `curl -L --fail -O https://www.eth3d.net/data/${scene}_dslr_undistorted.7z` and `..._scan_clean.7z`, extract with `7z x -y`. Scenes: `courtyard delivery_area facade`. Needs `apt install p7zip-full`. This is `$ETH3D_ROOT` (native), NOT the MoGe mono-depth bundle. |
| ETH3D 13-scene train (DA-V2 Table 2) | ~22 GB + eval extras | Staged ✅; harness OFF-PAPER (~−32 % under paper). **Parked** — return: [`docs/ETH3D_DAV2_TABLE2_HANDOFF.md`](docs/ETH3D_DAV2_TABLE2_HANDOFF.md) (D31/D33). Probe: `scripts/probe-eth3d-official-depth.py`. |
| ETH3D MoGe-eval bundle | 1.4 GB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'ETH3D*' --local-dir /tmp/moge && unzip /tmp/moge/ETH3D.zip -d $ETH3D_MOGE_ROOT` (nested `ETH3D/<scene>/<frame>/`, 453 samples; mono-depth, distinct from chamfer ETH3D). |
| DDAD MoGe-eval bundle | 0.6 GB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'DDAD*' --local-dir /tmp/moge && unzip /tmp/moge/DDAD.zip -d $DDAD_MOGE_ROOT` (1000 samples, 1400×700 warp; Tier-A Table 3). |
| Sintel MoGe-eval bundle | 0.5 GB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'Sintel*' --local-dir /tmp/moge && unzip /tmp/moge/Sintel.zip -d $SINTEL_MOGE_ROOT` (1064 frames, 872×436; MoGe Table 3 — NOT `$SINTEL_ROOT`). |
| Sintel (depth+cam+RGB) | 6 GB | See § "Sintel — NOT auth-gated" below. Native DA-V2 Table 2 **parked** (OFF-PAPER: ViT-L AbsRel **0.232** vs **0.487**; `clean` pass **0.222** — see [`docs/SINTEL_DAV2_TABLE2_HANDOFF.md`](docs/SINTEL_DAV2_TABLE2_HANDOFF.md)). Probe: `scripts/probe-sintel-pass.py`. |
| DTU MVS-22 | 7 GB | `gdown 135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_ -O dtu_test.zip && unzip dtu_test.zip` (MVSNet test, ~554 MB) + `aria2c -x 16 -s 16 https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip` (GT clouds, ~7 GB). **Do NOT fetch SampleSet.zip — that's a 2-scan demo.** |
| 7-Scenes | 12 GB | Per scene from `http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/${scene}.zip`, then unzip nested `seq-*.zip`. Scenes: `chess fire heads office pumpkin redkitchen stairs`. |

**Sintel — NOT actually auth-gated (corrected 2026-05-30).** The MPI-Sintel
depth + camera + image training archives are direct downloads, no
registration. Now staged on S3 (`datasets/sintel/`, has `training/{final,
clean,depth,camdata_left}`). If re-fetching:
```bash
curl -L -O https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip  # depth+camera
curl -L -O https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip                 # final+clean RGB
unzip -o '*.zip' -d $SINTEL_ROOT   # → training/{final,clean,depth,camdata_left}
```

**Auth-gated, deprioritized** (loaders work, data isn't on the v0.1 critical
path; substitutes already promoted in plan.md § 2):

| Dataset | Why deprioritized |
|---|---|
| ScanNet v2 / ScanNet-1500 | ToS at http://www.scan-net.org/, ≤24 h approval |

After a successful fetch, push the dataset back to S3 so the next
session inherits it:

```bash
aws s3 sync $<DATASET>_ROOT/ s3://plumbline-bench/datasets/<dataset>/ \
    --exclude '*/.plumbline_manifest/*'
```

**Disk budget gate.** Sum your queue's footprints + ~10 GB for model
weights + ~5 GB for the prediction cache + 20 % headroom. Compare to
the rental box's disk before booking. Standard RunPod / vast.ai NVMe is
100–200 GB so most queues fit, but Co3Dv2 or full ETH3D (50 GB) breaks
this — pin a sequence/scene whitelist before fetching.

### Model-weight gotchas (HF Hub)

- **Many HF repos ship both `model.pt` AND `model.safetensors`.** Use
  `--include 'model.safetensors' --include '*.json' --include 'README.md'`
  for selective fetch. `PyTorchModelHubMixin` (VGGT and similar) prefers
  `safetensors` regardless.
- **Diffusers-style repos (Marigold, GeoWizard) split weights across
  subfolders** (`unet/`, `vae/`, `text_encoder/`, `scheduler/`). Use
  `--include '*.safetensors' --include '*.json'` to pick all of them.
  Don't exclude `text_encoder` even if you think you don't need it.
- **After an interrupted `hf download`:** `find ~/.cache/huggingface/hub
  -name '*.incomplete' -delete` before retrying.
- `HF_HUB_DISABLE_XET=1` is set in pre-flight; keep it.

## Single-record diff workflow

Per plan.md § 12 — the *only* path to closing chamfer / off-paper
reproductions where the gap is non-trivial. Don't burn GPU hours on
full-dataset runs to discover a 130× discrepancy.

1. Pick one sample. Save its sample_id.
2. Clone the reference repo for the paper you're reproducing (e.g.
   CUT3R for D3/D4, Marigold for D9/D22).
3. Run reference and plumbline on the same sample. Save intermediate
   tensors at each stage as `.npy` (and a hash):
   - sample loading (raw bytes hash)
   - image preprocessing (post-resize / crop / normalize)
   - model input tensor
   - model output (raw, before postprocess)
   - postprocess (alignment, clamp, mask, units)
   - GT preprocessing
   - per-pixel error tensor (before metric aggregation)
4. Diff at each stage. The first diverging stage is the bug.
5. Fix the upstream stage *before* moving on — divergence compounds.
6. When stages 1–7 match within numerical tolerance, run the
   full-dataset reproduction.

If the reference repo doesn't exist or doesn't run, that's an
upstream-blocked issue (e.g. D22) — document and demote, don't guess.

## Failure handling for full-dataset runs

| Symptom | Action |
|---|---|
| `DatasetNotAvailable` | Skip; mark "data missing" in the report |
| `torch.cuda.OutOfMemoryError` | Halve `--max-views`; if still fails, mark "OOM" |
| Other exception | Capture full traceback; mark "failed"; continue |
| Exit 0 + paper_match true | ✅ MATCH |
| Exit 0 + paper_match false | ⚠️ off paper — record observed vs published, link to a D-number in `docs/DISCREPANCIES.md` |
| Exit 0 + paper_match null on a `verified_pdf` row | flag for human review (should not happen) |

**Never** modify YAMLs, edit `paper_reference.value`, edit tolerances,
or "fix" a failure by relaxing the protocol.

After each successful run, push the prediction cache back so the next
session can re-score under different alignment / metrics without
re-inferring:

```bash
aws s3 sync ~/.cache/plumbline/predictions/ s3://plumbline-bench/predictions/
```

Per-run results + logs go to `s3://plumbline-bench/runs/<ts>/`:

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p /tmp/results /tmp/logs
# ... runs write to those ...
aws s3 sync /tmp/results/ "s3://plumbline-bench/runs/${ts}/results/"
aws s3 sync /tmp/logs/ "s3://plumbline-bench/runs/${ts}/logs/"
```

## v0.1 gate status

Gate is **multi-cell**, replacing the original single-reproduction
gate (`plumbline reproduce vggt-paper-dtu-mvs`) that was retired
2026-04-27 when D3 hit upstream-block. See `plan.md` § 2.

- **≥ 15 verified_pdf paper-match cells** across ≥ 3 datasets and ≥ 5
  papers — **met 2026-04-27** (16 mono-depth cells across NYU + KITTI
  + DIODE).
- **≥ 1 pose paper-match** — *pending*. CO3Dv2 infra landed; GPU run
  pending for `vggt-co3dv2-pose` (VGGT Table 1, AUC@30 = 0.882) and
  `mast3r-co3dv2-pose` (MASt3R Table 3, mAA(30) = 0.818).
- **No fabricated paper cells** — every `verified_pdf` YAML audited
  against the source PDF (table + col + row). See
  `reproductions/AUDIT.md` for the per-YAML log. As of 2026-05-23 all
  25 `verified_pdf` YAMLs with a pinned value are PDF-confirmed
  (the last gap, `mast3r_co3dv2_pose`, was closed by a direct PDF read
  of MASt3R Table 3 — D23 resolved; the two GeoWizard paper targets
  were also audited for the first time and confirmed).

D3 (VGGT-DTU) and D4 (VGGT-ETH3D) are no longer gate items —
upstream-blocked / awaiting D10 respectively. See `REPRODUCTIONS.md`
for the live matrix.

## S3 cache layout

```
s3://plumbline-bench/
├── datasets/<name>/<sample_id>/*    # source datasets, selectable per sample
├── hf-cache/                        # HF model weights
├── torch-hub-cache/                 # Metric3D-v2 torch.hub
├── predictions/<model>/<hash>/<dataset>/   # cached predictions for re-scoring
└── runs/<ts>/                       # per-session results + logs + reports
```

**IAM:** bucket is `us-west-2`, SSE-S3 (AES-256), no versioning. User
`plumbline-gpu-cache` has GetObject + PutObject (no Delete). Session
token from `aws sts get-session-token --duration-seconds 43200` ≤12 h.

## Cost reference

| GPU | $/hr (vast) | $/hr (lambda) |
|---|---|---|
| 4090 | ~0.30 | N/A |
| A100 | ~0.80 | ~1.10 |
| H100 | ~1.80 | ~2.80 |

Typical session (5–10 reproductions on 4090): $1–3.

## Tear down

Don't delete the S3 cache. Don't terminate the rental box yourself —
leave it for the user. Stop work and report when:

- Every queued reproduction has been attempted (success or documented
  failure).
- Time / cost budget hit.
- Disk > 90 % full and no fetches succeeded recently.
- A pre-flight assumption broke (AWS session expired → ask the user
  to mint a new token).

## Known gotchas

- **ScanNet poses** can be `inf` on tracker-dropped frames; loader
  filters silently.
- **Scale alignment** must match the paper: DA-V2 (own Table 2) →
  `scale_shift`, Metric3Dv2 → `none`, MASt3R → `median`, VGGT → `none`,
  Marigold / GeoWizard → `scale_shift_depth`. **MoGe-eval mono-depth
  (DIODE/KITTI/GSO/iBims/ETH3D bundles) → `scale_shift_clamped`** — MoGe's
  own protocol (`moge/test/metrics.py`) floors aligned disparity at
  `1/gt.max()`. Plain `scale_shift` is fine on clean-indoor sets (iBims, NYU)
  but on any set with outdoor far-depth (ETH3D, DIODE, KITTI) a few pixels
  invert to enormous depths and a single sample blows up mean AbsRel
  (ETH3D-moge: mean 169 / median 0.0234 under scale_shift; 0.032 under
  clamped — see D30).
- **DA-V2 paper-match needs `$DAV2_ROOT`.** The adapter *defaults* to
  `source="paper"` (the `.pth` checkpoints — HF `-hf` re-exports score ~0.002
  lower and tip cells off-gate, see commit c14d776), which imports the model
  class from a clone of `github.com/DepthAnything/Depth-Anything-V2`. Despite
  the install table calling DA-V2 "base", every DA-V2 reproduction needs:
  `git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2
  $HOME/deps/depth-anything-v2 && export DAV2_ROOT=$HOME/deps/depth-anything-v2`
  (the `.pth` weights auto-download from HF). Without it, every sample is
  caught as "OOM or adapter error" and the run ends `n_evaluated=0`,
  `observed=nan` — looks like a silent non-result. **Always check
  `n_evaluated` in the report JSON.**
- **Depth vs disparity**: every adapter must emit depth in
  `Prediction.depth`. Check `metadata["native_space"]` to see what
  upstream emits before conversion.
- **CUDA nondeterminism**: some VGGT ops are nondeterministic on CUDA
  even with seeds. Use the YAML's `tolerance_relative`; don't chase
  bitwise match.
- **OOM**: runner catches and skips the sample; check
  `report.n_skipped > 0` at end.
- **HF rate limits**: if downloads stall, `hf auth login` first.
