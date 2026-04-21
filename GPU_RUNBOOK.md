# GPU runbook

Everything you need to run `plumbline` against real weights + real data on
a rented GPU. Optimized for a "spin up → run → tear down" workflow under
an hourly billing clock.

> **Running this autonomously?** If a Claude Code agent is driving the
> rental box (rather than a human in front of a terminal), point the
> agent at [`docs/AGENT_GPU_RUNBOOK.md`](./docs/AGENT_GPU_RUNBOOK.md)
> instead. That doc is denser, ordered for execution, and explicit
> about what the agent must NOT do.

## Provider-agnostic box setup

Two paths:

### Path A — Docker (recommended for fresh boxes)

If the GPU host has Docker + `nvidia-container-toolkit`:

```bash
git clone git@github.com:kmatzen/plumbline.git && cd plumbline

# Build once; slim image (no research repos)
docker build -t plumbline .

# Or full image with VGGT baked in
docker build --build-arg WITH_GIT_DEPS=1 -t plumbline:full .

# Verify GPU visibility
docker run --rm --gpus all plumbline --help

# Run a reproduction, mounting host data + cache
docker run --rm --gpus all \
    -v $HOME/data:/data \
    -v $HOME/.cache/plumbline-docker:/cache \
    plumbline reproduce da-v2-small-nyuv2
```

Caches persist across runs via the `-v $HOME/.cache/plumbline-docker:/cache`
mount (holds both the HF/torch weight cache and plumbline's prediction
cache). Re-running the same reproduction is essentially free after first
inference.

### Path B — Native install

On a fresh Ubuntu 22.04 box with CUDA drivers preinstalled (Lambda,
vast.ai, RunPod, etc.):

```bash
# 1. Tools
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone + install (private repo; supply a deploy key or token)
git clone git@github.com:kmatzen/plumbline.git
cd plumbline
uv sync --extra models

# 3. Verify torch sees the GPU
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 4. Verify the console script works
uv run plumbline list-models
```

## S3 cache (`plumbline-bench` bucket, us-west-2)

Slow-to-fetch datasets (DTU, KITTI pruned per-drive subsets,
etc.) are cached in a personal S3 bucket so each new rental box can
rehydrate from S3 rather than redoing the hours of upstream downloads.
Storage footprint is small (~15-20 GB); the bucket is scoped to one
IAM user with least-privilege access.

### Once, on your laptop

Configure an AWS profile with the long-lived IAM-user credentials
(from the `plumbline-gpu-cache` user; see `infrastructure notes`
below for how it was provisioned):

```bash
aws configure --profile plumbline-gpu-cache
# AWS Access Key ID:     AKIA...
# AWS Secret Access Key: ...
# Default region name:   us-west-2
# Default output format: json
```

### Before each rental

Mint a 12-hour session token:

```bash
scripts/gpu_box_session_token.sh
```

…which prints `export` lines. Paste them into the rental box's shell.

### On the rental box

```bash
# After pasting the exported session creds:
aws s3 ls s3://plumbline-bench/                 # smoke test
aws s3 sync s3://plumbline-bench/datasets/ ~/data/   # pull cached datasets
# ... run reproductions ...
aws s3 sync ~/data/ s3://plumbline-bench/datasets/ --exclude '*/.plumbline_manifest/*'  # push fresh ones back
```

### Infrastructure notes

- Bucket `plumbline-bench` — `us-west-2`, SSE-S3 default encryption
  (AES-256), no versioning (cache; overwrite is the intent).
- IAM policy `plumbline-cache-access` (customer-managed) grants only:
  - `s3:ListBucket`, `s3:ListBucketMultipartUploads` on the bucket
  - `s3:GetObject`, `s3:PutObject`, `s3:AbortMultipartUpload`,
    `s3:ListMultipartUploadParts` on `plumbline-bench/*`
  - **No delete.** Remove stale objects from your laptop with root
    creds if you need to.
- IAM user `plumbline-gpu-cache` — policy above attached, programmatic
  access only (no console login).
- Session-token shape (`aws sts get-session-token --duration-seconds 43200`)
  means leaked creds on the rental box auto-expire in ≤12h.

## HuggingFace login (for DA-V2, DA3, Metric3Dv2)

Rate-limited downloads for anonymous users. Log in once:

```bash
export HF_HOME="$HOME/.cache/huggingface"   # optional; point at fast disk
uv run huggingface-cli login
```

## Dataset downloads

### NYUv2 (public, single .mat, ~3 GB)

The primary mono-depth benchmark. The "labeled" subset contains 1449
RGB-D pairs; the 654-sample Eigen test split is the canonical eval.

```bash
mkdir -p ~/data/nyuv2 && cd ~/data/nyuv2
curl -L -O https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
export NYUV2_ROOT=$HOME/data/nyuv2
```

``NYUv2Dataset`` defaults to ``depth_field="raw"`` (the sparse Kinect
measurements, with holes) — this is what the Eigen 2014 protocol uses
and what every DA-V2 / Metric3Dv2 paper number in the status matrix
targets. Pass ``depth_field="filled"`` for Silberman's colorization-
filled variant.

### KITTI (public, Eigen-benchmark 652-frame subset)

KITTI raw is ~65 GB across 28 drives if you fetch every frame — but
the Eigen benchmark only evaluates 12–25 frames per drive (652
total). The committed sample list
(`reproductions/kitti_eigen_benchmark_652.txt`) drives a selective
fetcher that downloads each drive zip, extracts only the listed
``image_02`` PNGs + calib, and discards the rest:

```bash
export KITTI_ROOT=$HOME/data/kitti
scripts/fetch_kitti.py --kitti-root $KITTI_ROOT
```

Bandwidth: ~8.5 GB (28 drive zips × ~300 MB). On-disk footprint
after prune: ~700 MB for the 652 RGB frames + calib. Resumable —
re-running skips drives whose listed frames are already present.

Annotated-depth GT is a separate bundle (single 14 GB zip covering
all drives); fetch once, unpack once:

```bash
curl -L -o /tmp/data_depth_annotated.zip \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
mkdir -p $KITTI_ROOT/depth_annotated
unzip -d $KITTI_ROOT/depth_annotated /tmp/data_depth_annotated.zip
rm /tmp/data_depth_annotated.zip
```

After the fetch, push the pruned KITTI tree to the S3 cache so
future rentals skip the 8.5 GB re-download:

```bash
aws s3 sync $KITTI_ROOT s3://plumbline-bench/datasets/kitti/ \
    --exclude '*/.plumbline_manifest/*'
```

### Sintel (public images; depth + cameras are auth-gated)

**The optical-flow bundle is public** (~5.3 GB) and gives you the `final/`,
`clean/`, and flow data. It does NOT contain depth or camera archives:

```bash
mkdir -p ~/data/sintel && cd ~/data/sintel
curl -L -o complete.zip http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip complete.zip && rm complete.zip
export SINTEL_ROOT=$HOME/data/sintel
```

**Depth + camera archives require registration** at
https://sintel.is.tue.mpg.de/signup. After login, download the Depth +
Camera Motion archive from https://sintel.is.tue.mpg.de/depth and extract
into the same `$SINTEL_ROOT` so the layout merges to:

```
$SINTEL_ROOT/training/
  final/<scene>/frame_XXXX.png        # from complete.zip (public)
  clean/<scene>/frame_XXXX.png        # from complete.zip (public)
  depth/<scene>/frame_XXXX.dpt        # from depth archive (auth)
  camdata_left/<scene>/frame_XXXX.cam # from depth archive (auth)
```

The `plumbline` Sintel loader raises `DatasetNotAvailable` with a pointer
to the above if any of these are missing. `tests/test_real_imagery_integration.py`
runs against `final/` only (no auth) and gives a real-data smoke check for
the monocular adapters.

### ScanNet v2 test (auth-gated)

1. Sign the Terms of Use at http://www.scan-net.org/ (≤24h approval).
2. Run the official download script with `--type scans_test`.
3. Unpack `.sens` files to `color/*.jpg`, `depth/*.png`, `pose/*.txt`,
   `intrinsic/` using the `SensReader` from the ScanNet repo.
4. Set `export SCANNET_ROOT=<path>`.

### ETH3D high-res multi-view (public, no auth; ~50 GB for all 13 train scenes)

Needs `p7zip-full` for the archives:

```bash
sudo apt-get install -y p7zip-full
```

Per-scene, we want the **undistorted** archive (images + calibration; the
loader expects ``dslr_calibration_undistorted/cameras.txt`` etc.) and the
**scan eval** archive (laser-scan GT):

```bash
mkdir -p ~/data/eth3d && cd ~/data/eth3d
# Training scenes w/ GT (the only ones we eval on):
for scene in courtyard delivery_area electro facade kicker meadow \
             office pipes playground relief relief_2 terrace terrains; do
  curl -L --fail -O "https://www.eth3d.net/data/${scene}_dslr_undistorted.7z"
  curl -L --fail -O "https://www.eth3d.net/data/${scene}_dslr_scan_eval.7z"
  7z x -y "${scene}_dslr_undistorted.7z"
  7z x -y "${scene}_dslr_scan_eval.7z"
done
export ETH3D_ROOT=$HOME/data/eth3d
```

Archive names to know:
- `*_dslr_undistorted.7z` — undistorted jpg + COLMAP calibration
- `*_dslr_jpg.7z` — distorted jpg only (don't need for eval)
- `*_dslr_scan_eval.7z` — laser GT in the evaluation coordinate frame
- `*_scan_clean.7z` — the raw clean laser scan (`.ply`)

### iBims-1 (MoGe preprocessed bundle, public)

100 high-fidelity indoor scenes (Koch et al. 2018) rendered into the
same per-scene directory format as GSO: `image.jpg` +
log-encoded uint16 `depth.png` + `segmentation.png` + normalised-K
`meta.json`. Bundle is tiny (~40 MB zipped); perfect "quick high-quality
indoor" slot for MoGe Table 1/2 reproductions.

```bash
pip install huggingface-hub
hf download Ruicheng/monocular-geometry-evaluation \
    --repo-type dataset --include 'iBims-1*' --local-dir ~/data/moge_eval
cd ~/data/moge_eval && unzip iBims-1.zip
export IBIMS1_ROOT=$HOME/data/moge_eval/iBims-1
```

The TUM upstream release (https://www.asg.ed.tum.de/lmf/ibims1/) is
the canonical source but ships a .mat-based layout that this loader
does not speak. Stick with the MoGe bundle.

### DTU (public, no ToS; ~7.5 GB across two archives)

The full 22-scan MVS test set needs TWO archives. Do NOT confuse
either with DTU's `SampleSet.zip` (6.9 GB on the same server) —
that's a format-demo with scans 1 & 6 only, not the eval set.

```bash
mkdir -p ~/data/dtu && cd ~/data/dtu

# 1. MVSNet preprocessed test set (~554 MB): images + cameras + pair.txt
#    across all 22 test scans. Google Drive, public. Uses gdown.
pip install gdown
gdown "135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_" -O dtu_test.zip
unzip dtu_test.zip && rm dtu_test.zip

# 2. DTU GT point clouds (~6.97 GB): Points/stl/stl*_total.ply per scan.
#    Use aria2 for parallel ranges (single-stream from this server caps
#    at ~680 KB/s; aria2 -x 16 reaches 8 MB/s).
brew install aria2   # if not already installed
aria2c --max-connection-per-server=16 --split=16 --min-split-size=1M \
    --out Points.zip \
    "https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
unzip Points.zip && rm Points.zip

export DTU_ROOT=$HOME/data/dtu
```

Resulting layout matches `DTUDataset`'s auto-detected test format:

```
$DTU_ROOT/dtu/scan1/{cams,images,pair.txt}
$DTU_ROOT/dtu/scan4/...
...
$DTU_ROOT/Points/stl/stl001_total.ply
$DTU_ROOT/Points/stl/stl004_total.ply
...
```

### 7-Scenes (Microsoft, public, no auth; ~12 GB for all 7 scenes)

RGB-D + per-frame pose, Kinect v1. Default test-split per Shotton 2013
(loaded automatically by `SevenScenesDataset(split="test")`). Download
per-scene zips; each is ~1-2 GB after unpacking.

```bash
mkdir -p ~/data/7scenes && cd ~/data/7scenes
base="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
for scene in chess fire heads office pumpkin redkitchen stairs; do
  curl -L --fail -O "${base}/${scene}.zip"
  unzip "${scene}.zip" && rm "${scene}.zip"
  # Scenes unpack to nested zips per sequence; extract the inner ones too.
  (cd "$scene" && for f in seq-*.zip; do unzip -o "$f" && rm "$f"; done)
done
export SEVEN_SCENES_ROOT=$HOME/data/7scenes
```

Loader options:
- `split="test"` uses the canonical test-sequences-per-scene split.
- `views_per_sample=2` (default) yields two-view pairs for relative-
  pose evaluation; MASt3R paper §4.2 uses this shape.
- `stride=10, baseline=10` is the default — window starts every 10
  frames, pair spans 10 frames (≈0.3 s at 30 fps). Tune per paper.

## Per-adapter first-run notes

### DepthAnything V2 (fully wired)

Uses HuggingFace Transformers. No extra setup. Six variants:
`small`, `base`, `large`, `metric-indoor-{small,base,large}`,
`metric-outdoor-{small,base,large}`.

Paper-match run on NYUv2 (all three relative variants land inside
their tolerance bands):

```bash
export NYUV2_ROOT=/path/to/nyuv2
uv run plumbline reproduce da-v2-small-nyuv2   # AbsRel 0.053 → 0.051
uv run plumbline reproduce da-v2-base-nyuv2    # AbsRel 0.049 → 0.046
uv run plumbline reproduce da-v2-large-nyuv2   # AbsRel 0.045 → 0.043
```

Or a custom run:

```bash
uv run plumbline run --model depth-anything-v2 \
  --dataset nyuv2 --tasks mono_depth \
  --scale-alignment scale_shift --max-views 1 \
  -o da-v2_nyuv2.json
```

Sintel with depth GT is still gated — see the dataset section above.

### Metric3Dv2 (torch.hub)

Pulls from https://github.com/YvanYin/Metric3D via `torch.hub.load`
with `trust_repo=True` on first use; weights download from the
JUGGHM/Metric3D HuggingFace mirror. Upstream's hubconf depends on
`mmengine` and imports through `mmcv` (with a fallback). Install the
pure-Python variants:

```bash
VIRTUAL_ENV=.venv uv pip install mmengine mmcv-lite
```

**Important env hygiene:** if `xformers` is installed with a prebuilt
wheel that doesn't match your torch/CUDA exactly, Metric3D's bundled
dinov2 backbone will `import xformers.ops` successfully but then raise
`NotImplementedError` inside `memory_efficient_attention_forward` at
forward time. Two options:

1. Ensure a matching xformers wheel is installed for your torch/CUDA.
2. Uninstall xformers — Metric3D falls back to a correct pure-PyTorch
   attention path. This is what plumbline's CI does:

   ```bash
   VIRTUAL_ENV=.venv uv pip uninstall xformers
   ```

Known upstream gotcha wired into the adapter: `torch.hub.load`'s
pretrained-weight flag is `pretrain=True` (no "ed") — passing
`pretrained=True` is silently swallowed by `**kwargs` so the model
returns NaN on the first forward. `Metric3Dv2Adapter` passes the right
spelling; if you write your own integration, don't trip on this.

Canonical-camera protocol (replicated in the adapter):

- `cv2.resize` to fit (616, 1064) at native aspect ratio; scale
  intrinsics by the same factor.
- Pad with ImageNet mean colour to exactly (616, 1064).
- Normalise with ImageNet mean/std at **[0, 255] scale** (not [0, 1]).
- `model.inference({'input': ...})` → `(pred_depth, conf, out)`.
- Un-pad, upsample to native, multiply by `scaled_fx / 1000` to
  de-canonicalise to metric meters.

Metric3Dv2 is metric; reproductions use `scale_alignment: none`.
Paper matches ViT-L and ViT-Giant2 on NYU land inside ±10% — see
`reproductions/metric3d_v2_nyuv2.yaml` and `..._giant_nyuv2.yaml`.

### MASt3R (upstream not on PyPI — or pip-installable)

Upstream has no `pyproject.toml`; needs a recursive clone so the `dust3r`
and `croco` submodules are available. Install transitive deps into the
plumbline venv:

```bash
git clone --recursive https://github.com/naver/mast3r /workspace/deps/mast3r
VIRTUAL_ENV=/workspace/plumbline/.venv uv pip install roma scikit-learn trimesh
export MAST3R_ROOT=/workspace/deps/mast3r   # DUST3R_ROOT defaults to $MAST3R_ROOT/dust3r
```

`MASt3RAdapter` lazy-adds `$MAST3R_ROOT` + `$DUST3R_ROOT` to `sys.path` on
first use, runs `dust3r.inference.inference` on a symmetrized pair, then
fits a `PairViewer` global-aligner to recover per-view focal, principal
point, camera-frame depth, and world-from-camera pose. The adapter rebases
the scene so view 0 is always the world frame (PairViewer may pick either
view as origin based on pair-confidence).

v0.1 caps `max_views=2`. N>2 requires `mast3r.cloud_opt.sparse_ga.
sparse_global_alignment`, which is iterative — a v0.2 item.

### VGGT (upstream pip-installable from git)

```bash
VIRTUAL_ENV=/workspace/plumbline/.venv uv pip install git+https://github.com/facebookresearch/vggt
```

`VGGTAdapter` already wires `_run_vggt` against the upstream
`vggt.models.vggt.VGGT` + `vggt.utils.pose_enc.pose_encoding_to_extri_intri`
API. Preprocessing mirrors upstream's `mode="crop"`: width=518, height
rounded to the nearest multiple of 14 preserving aspect ratio, centre-crop
if tall. Autocast defaults to bf16 on CUDA (sm≥8, matching the VGGT demo);
set `dtype: float32` in the reproduction YAML for a debug run.

Extrinsic convention: `pose_encoding_to_extri_intri` returns
`camera_from_world`; the adapter inverts to `world_from_camera` and
rebases to make view 0 identity (VGGT's pose encoding already biases
view 0 toward identity within float noise, so this is usually a no-op).

VGGT at 32 views ×1024² fits in 24 GB (A100 / 4090). An RTX 3090 handles
8 views at the 518-default width comfortably under 10 GB. OOM? the runner
catches and skips the sample; drop `--max-views` to 4 or adjust the YAML.

### Depth Anything 3 (pip package)

```bash
VIRTUAL_ENV=.venv uv pip install depth-anything-3
```

Default checkpoint is `depth-anything/DA3-LARGE` (the adapter also
handles `DA3-LARGE-1.1`, the bug-fix release, and `DA3MONO-LARGE`).
Preprocessing is DA3's own canonical-camera path with a 504-pixel long
edge; the adapter feeds `model.inference(images, export_dir=None)` and
harvests depth / conf / extrinsics / intrinsics.

Extrinsic convention: DA3's `Prediction.extrinsics` are `w2c` 3x4
(camera-from-world). The adapter pads to 4x4 and inverts to
world-from-camera; a rebase guard handles the (rare) case where view 0
isn't identity out of the box.

DA3's public mono-depth checkpoints are **relative** on NYU (pred /
GT ratio ≈ 0.32 unaligned). Use `scale_alignment: scale_shift` for
NYU — `reproductions/da3_nyuv2.yaml` matches the paper's Table 4 δ₁.
The "DA3-metric" checkpoint referenced in the paper's Table 11 is not
publicly released; don't expect the metric AbsRel=0.070 to be
reproducible until it ships.

### π³ (Pi-Cubed) — upstream not on PyPI; clone + sys.path

ByteDance's multi-view 3D foundation model (ICLR 2026 submission).
Same install pattern as MASt3R / GeoWizard: clone the repo, point
`$PI3_ROOT` at it; the adapter lazy-imports
`pi3.models.pi3{,x}.{Pi3,Pi3X}`. Two checkpoints on HF:
`yyfz233/Pi3` (original) and `yyfz233/Pi3X` (Dec 2025 improved rev,
adapter default).

```bash
git clone https://github.com/yyfz/Pi3 /workspace/deps/pi3
cd /workspace/deps/pi3 && uv pip install -r requirements.txt
export PI3_ROOT=/workspace/deps/pi3
```

Input/output convention matches plumbline:
- Input: `(N, H, W, 3)` uint8 sRGB at native resolution. Adapter
  normalises to `[0, 1]` and reshapes to `(1, N, 3, H, W)`.
- Output: `local_points[..., 2]` → per-view depth (z); `camera_poses`
  → world_from_camera (already OpenCV); `points` → world-frame point
  map; `conf` → per-pixel confidence after sigmoid.
- Default `dtype: bfloat16`. Defaults `max_views: 16` (conservative;
  can raise on >24 GB cards).

Smoke test (no paper-row YAML committed yet — first-run results pin
the value):

```bash
uv run plumbline run --model pi3 --dataset eth3d \
    --tasks mvs_depth pose --max-views 8 -o pi3_eth3d.json
```

### GeoWizard (upstream not on PyPI — clone + sys.path)

Upstream ships as a CLI-oriented repo without a Python package
interface. The adapter lazy-imports
`models.geowizard_pipeline.DepthNormalEstimationPipeline` from a local
clone pointed at by `$GEOWIZARD_ROOT`. Weights come from
`lemonaddie/Geowizard` on HuggingFace (~5 GB).

```bash
git clone https://github.com/fuxiao0719/GeoWizard /workspace/deps/geowizard
export GEOWIZARD_ROOT=/workspace/deps/geowizard
# Extra deps that upstream pins but doesn't ship as a requirements.txt in the
# `geowizard/` subdir — install into plumbline's venv:
VIRTUAL_ENV=/workspace/plumbline/.venv uv pip install accelerate transformers
```

Domain conditioning: GeoWizard is a single model with per-sample
`domain={"indoor","outdoor","object"}` conditioning. The reproduction
YAML picks the domain to match the paper's target table — indoor for
NYU / DIODE-indoor / iBims, outdoor for KITTI / DIODE-outdoor, object
for GSO.

Alignment: GeoWizard outputs affine-invariant depth in [0, 1] like
Marigold. Use `scale_alignment: scale_shift_depth` (fit scale+shift in
depth space) rather than the inverse-depth fit DA-V2 / MoGe use.

Smoke test (no paper-row YAML yet — add after first run pins a
reference value):

```bash
uv run plumbline run --model geowizard --dataset nyuv2 \
    --tasks mono_depth --scale-alignment scale_shift_depth \
    --max-views 1 -o geowizard_nyuv2.json
```

## v0.1 gate status

The originally-planned gate is
`plumbline reproduce vggt-paper-scannet-depth` — VGGT wiring is
complete and sanity-checked end-to-end on random and ETH3D courtyard
inputs, but the reproduction itself is blocked on ScanNet ToS signup
and data. When the user has `$SCANNET_ROOT` set, the steps are:

```bash
uv run plumbline reproduce vggt-paper-scannet-depth -o vggt_scannet.json
```

The YAML at `reproductions/vggt_scannet_depth.yaml` has
`paper_reference.value = 0.0` as a placeholder. On the first successful
run:

1. Record the observed AbsRel in the YAML.
2. Pin the sample list:
   ```bash
   uv run plumbline make-samples \
     --dataset scannet --data-root $SCANNET_ROOT \
     --split test --subset 100 \
     -o reproductions/vggt_scannet_depth.samples.txt
   ```
   Add `sample_ids_file: vggt_scannet_depth.samples.txt` to the YAML
   (remove the numeric `subset:` field). Freezes the sample set across
   manifest re-scans.
3. Set `tolerance_relative` ±5% (widen if CUDA nondeterminism pushes it).
4. Update the table in [REPRODUCTIONS.md](./REPRODUCTIONS.md).

In the meantime, the plumbline has a **7-row paper-match matrix** on
NYUv2 (DA-V2 S/B/L, DA-V2 Metric-Indoor-L, Metric3Dv2 L/Giant2, DA3)
that confirms the pipeline end-to-end for mono depth. See
[REPRODUCTIONS.md](./REPRODUCTIONS.md) for the current status.

## Cost tracking

Per `plan.md § 7`, the v0.1 budget is ~$40 compute. Typical rates
(as of 2026-04):

| GPU   | $/hr (vast) | $/hr (lambda) |
| ---   | ---         | ---           |
| 4090  | ~0.30       | N/A           |
| A100  | ~0.80       | ~1.10         |
| H100  | ~1.80       | ~2.80         |

DA-V2 + Sintel shakedown: 4090 × 1 h. VGGT ScanNet full eval: A100 × 10 h.

## Gotchas

- **ScanNet poses** have `inf` values on tracker-dropped frames; loader
  filters these silently.
- **Scale alignment** must match the paper: DA-V2 → `scale_shift`,
  Metric3Dv2 → `none`, MASt3R → `median`, VGGT → `none`.
- **Depth vs disparity**: every adapter's output must be depth in
  `Prediction.depth`, not disparity. Check `metadata["native_space"]` to
  see what upstream emits.
- **CUDA nondeterminism**: some VGGT ops are nondeterministic on CUDA
  even with seeds. Document the tolerance; don't chase bitwise match.
- **OOM**: the runner catches and skips the sample; check
  `report.n_skipped > 0` at the end and investigate if non-zero.
- **HF rate limits**: if downloads stall, `huggingface-cli login` first.
