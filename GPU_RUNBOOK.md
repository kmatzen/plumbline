# GPU runbook

Everything you need to run `plumbline` against real weights + real data on
a rented GPU. Optimized for a "spin up → run → tear down" workflow under
an hourly billing clock.

## Provider-agnostic box setup

These steps assume a fresh Ubuntu 22.04 box with CUDA drivers preinstalled
(Lambda, vast.ai, RunPod, etc.).

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

## HuggingFace login (for DA-V2, DA3, Metric3Dv2)

Rate-limited downloads for anonymous users. Log in once:

```bash
export HF_HOME="$HOME/.cache/huggingface"   # optional; point at fast disk
uv run huggingface-cli login
```

## Dataset downloads

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

### ETH3D high-res multi-view (public, ~50 GB for all scenes)

```bash
mkdir -p ~/data/eth3d && cd ~/data/eth3d
# Training scenes w/ GT (the only ones we eval on):
for scene in courtyard delivery_area electro facade kicker meadow \
             office pipes playground relief relief_2 terrace terrains; do
  curl -L -O "https://www.eth3d.net/data/${scene}_dslr_jpg.7z"
  curl -L -O "https://www.eth3d.net/data/${scene}_dslr_calibration_undistorted.7z"
  curl -L -O "https://www.eth3d.net/data/${scene}_dslr_scan_eval.7z"
  7z x "${scene}_dslr_jpg.7z"
  7z x "${scene}_dslr_calibration_undistorted.7z"
  7z x "${scene}_dslr_scan_eval.7z"
done
export ETH3D_ROOT=$HOME/data/eth3d
```

## Per-adapter first-run notes

### DepthAnything V2 (fully wired)

Uses HuggingFace Transformers. No extra setup.

```bash
uv run plumbline run --model depth-anything-v2 \
  --dataset sintel --tasks mono_depth \
  --scale-alignment scale_shift \
  --subset 50 \
  -o da-v2_sintel.json
```

Expected: AbsRel ~0.07–0.10 on Sintel with `scale_shift` alignment.

### Metric3Dv2 (torch.hub)

Pulls from https://github.com/YvanYin/Metric3D. On first run, torch.hub
clones the repo into `~/.cache/torch/hub/` and downloads weights. If
the entry-point name has drifted upstream, edit
`src/plumbline/models/metric3d_v2.py::_HUB_MODELS` accordingly.

```bash
uv run plumbline run --model metric3d-v2 \
  --dataset scannet --tasks mono_depth \
  --scale-alignment none \
  --data-root $SCANNET_ROOT \
  --subset 100 \
  -o metric3d_scannet.json
```

Metric3Dv2 is metric; use `--scale-alignment none`.

### MASt3R (upstream not on PyPI)

```bash
uv pip install git+https://github.com/naver/mast3r
```

Then **wire up** `_run_mast3r` in `src/plumbline/models/mast3r.py` per the
module docstring's output contract. The upstream API has moved around; do
a single-pair run from a scratch Python REPL first to confirm the shapes,
then transcribe into `_run_mast3r`.

### VGGT (upstream not on PyPI)

```bash
uv pip install git+https://github.com/facebookresearch/vggt
```

Then wire `_run_vggt` in `src/plumbline/models/vggt.py`. VGGT at 32 views
×1024² fits in 24 GB (A100 / 4090). OOM? the runner catches and skips the
sample; drop `--max-views` to 8 or 16.

### Depth Anything 3 (wiring TBD at release time)

Wiring is gated on upstream API stabilizing. Check the HF Hub model card
for the DA3 release name and update `DepthAnything3Adapter.checkpoint`.

## The v0.1 gate

```bash
uv run plumbline reproduce vggt-paper-scannet-depth -o vggt_scannet.json
```

The YAML config at `reproductions/vggt_scannet_depth.yaml` currently has
a placeholder `paper_reference.value = 0.0`. On the first successful run:

1. Record the observed AbsRel in `reproductions/vggt_scannet_depth.yaml`.
2. Pin the exact sample list:

   ```bash
   uv run plumbline make-samples \
     --dataset scannet --data-root $SCANNET_ROOT \
     --split test --subset 100 \
     -o reproductions/vggt_scannet_depth.samples.txt
   ```

   Then add `sample_ids_file: vggt_scannet_depth.samples.txt` to the
   YAML (and remove the numeric `subset:` field if present). This
   freezes the sample set so future runs don't drift when manifests
   change.
3. Set `tolerance_relative` to match VGGT paper ± 5% (adjust up if CUDA
   nondeterminism pushes it).
4. Update the table in [REPRODUCTIONS.md](./REPRODUCTIONS.md).

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
