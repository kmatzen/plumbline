# GPU runbook

Spin up → bring up → run → tear down. Single doc; the agent and the
human see the same instructions.

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
uv sync --extra models
uv run python -c "import torch; assert torch.cuda.is_available()"
uv run plumbline list-models
```

Set dataset-root env vars in `~/.bashrc-plumbline` for what you actually
need. Don't pre-create directories for datasets you won't touch.

```bash
export NYUV2_ROOT=$HOME/data/nyuv2
export KITTI_ROOT=$HOME/data/kitti
export ETH3D_ROOT=$HOME/data/eth3d
export DTU_ROOT=$HOME/data/dtu
# ... only add what the current session needs
```

### Per-adapter extras (install only what you need)

`--extra models` covers torch/diffusers/transformers — enough for DA-V2,
DA3, Marigold, Depth Pro, Metric3Dv2.

| Adapter | Extra setup |
|---|---|
| MoGe | `uv pip install 'git+https://github.com/microsoft/MoGe.git'` |
| VGGT | `uv pip install 'git+https://github.com/facebookresearch/vggt'` |
| MASt3R | `git clone --recursive https://github.com/naver/mast3r $HOME/deps/mast3r; uv pip install roma scikit-learn trimesh; export MAST3R_ROOT=$HOME/deps/mast3r` |
| GeoWizard | `git clone --depth 1 https://github.com/fuxiao0719/GeoWizard $HOME/deps/geowizard; export GEOWIZARD_ROOT=$HOME/deps/geowizard; uv sync --extra geowizard; uv pip install --force-reinstall 'nvidia-cudnn-cu12==9.1.0.70'` |
| π³ | `git clone https://github.com/yyfz/Pi3 $HOME/deps/pi3; cd $HOME/deps/pi3 && uv pip install -r requirements.txt; export PI3_ROOT=$HOME/deps/pi3` |

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

| Dataset | Min viable | Fetch |
|---|---|---|
| NYUv2 | 3 GB | `curl -L -O https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat` to `$NYUV2_ROOT` |
| KITTI Eigen-652 | 6 GB | `scripts/fetch_kitti.py --kitti-root $KITTI_ROOT` (selective per-drive) + `data_depth_annotated.zip` (~14 GB unpacked, but loader only needs the 652 listed frames) |
| KITTI MoGe-eval bundle | 3 GB | `hf download Ruicheng/monocular-geometry-evaluation KITTI.zip --repo-type dataset --local-dir /tmp/moge_dl && unzip /tmp/moge_dl/KITTI.zip -d $KITTI_MOGE_ROOT` |
| iBims-1 | 40 MB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'iBims-1.zip' --local-dir /tmp/moge && unzip /tmp/moge/iBims-1.zip -d $IBIMS1_ROOT/..` |
| GSO | 2 GB | `hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --include 'GSO.zip' --local-dir /tmp/moge && unzip /tmp/moge/GSO.zip -d $GSO_ROOT/..` |
| DIODE val | 2.8 GB | `curl -L -O http://diode-dataset.s3.amazonaws.com/val.tar.gz && tar xzf val.tar.gz -C $DIODE_ROOT/..` |
| ETH3D 3-scene | 8 GB | Per scene: `curl -L --fail -O https://www.eth3d.net/data/${scene}_dslr_undistorted.7z` and `..._dslr_scan_eval.7z`, extract with `7z x -y`. Scenes: `courtyard delivery_area facade`. Needs `apt install p7zip-full`. |
| DTU MVS-22 | 7 GB | `gdown 135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_ -O dtu_test.zip && unzip dtu_test.zip` (MVSNet test, ~554 MB) + `aria2c -x 16 -s 16 https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip` (GT clouds, ~7 GB). **Do NOT fetch SampleSet.zip — that's a 2-scan demo.** |
| 7-Scenes | 12 GB | Per scene from `http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/${scene}.zip`, then unzip nested `seq-*.zip`. Scenes: `chess fire heads office pumpkin redkitchen stairs`. |

**Auth-gated, deprioritized 2026-04-19** (loaders work, data isn't on
the v0.1 critical path; substitutes already promoted in plan.md § 2):

| Dataset | Why deprioritized |
|---|---|
| Sintel depth + cameras | Registration at https://sintel.is.tue.mpg.de/signup |
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

`plumbline reproduce vggt-paper-dtu-mvs` (chamfer 0.382 m, VGGT Table
2). Currently blocked by D3 (per-view-masked vs scene-merged metric
shape) — see `docs/DISCREPANCIES.md` and use the single-record diff
workflow against CUT3R `eval/mv_recon/`.

13 ✅ mono-depth cells (NYU + KITTI) are the de facto gate while D3 is
open. See `REPRODUCTIONS.md` for the live status matrix.

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
- **Scale alignment** must match the paper: DA-V2 → `scale_shift`,
  Metric3Dv2 → `none`, MASt3R → `median`, VGGT → `none`, Marigold /
  GeoWizard → `scale_shift_depth`.
- **Depth vs disparity**: every adapter must emit depth in
  `Prediction.depth`. Check `metadata["native_space"]` to see what
  upstream emits before conversion.
- **CUDA nondeterminism**: some VGGT ops are nondeterministic on CUDA
  even with seeds. Use the YAML's `tolerance_relative`; don't chase
  bitwise match.
- **OOM**: runner catches and skips the sample; check
  `report.n_skipped > 0` at end.
- **HF rate limits**: if downloads stall, `hf auth login` first.
