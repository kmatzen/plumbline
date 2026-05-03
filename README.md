# plumbline

A reproducible evaluation harness for 3D geometric foundation models —
think `lm-evaluation-harness`, but for models like **VGGT**, **Depth
Anything 3**, **MASt3R**, **Metric3Dv2**, and **Depth Anything V2**.

**Status:** v0.1 in development. 7 paper-match reproductions passing on
NYUv2; ETH3D pose sweeps across 2 models at multiple view counts. API
will still change before 1.0.

## What works today

- **6 model adapters** fully wired: DA-V2 (6 variants including metric
  Indoor/Outdoor), DA3, Metric3Dv2 (ViT-S/L/Giant2), MASt3R (PairViewer,
  2-view), VGGT, MoGe (v1 ViT-L + v2 variants including `*-normal`).
- **7 datasets**: NYUv2 (Eigen 2014 protocol, rawDepths), ETH3D high-res
  multi-view, Sintel (RGB + flow; depth archive still gated), ScanNet
  (loader ready; data still gated), KITTI (Eigen split against annotated
  GT — loader + Garg/Eigen crops ready), DIODE (indoor + outdoor dense
  RGB-D — loader with configurable intrinsic, bool mask → depth_valid),
  DTU (MVSNet-repacked, 22-scan MVS test split — the v0.1 paper-match
  target).
- **7 paper-match NYUv2 reproductions** — see [REPRODUCTIONS.md](./REPRODUCTIONS.md).
- **Pose metrics** include both absolute per-view and **pairwise
  relative-pose AUC** (the aggregation VGGT / MASt3R / DUSt3R papers
  report).
- **7-DoF similarity alignment** (Umeyama via camera-centre correspondences)
  wired through the runner for ETH3D / T&T / DTU chamfer protocols.

## Install

```bash
uv sync                       # from the repo
pip install plumbline-bench   # from PyPI
```

The base install pulls torch + the HF stack. CUDA flavor is whatever
your pip resolves (PyPI default: cu124 on Linux, CPU/MPS on macOS); to
override, install your preferred torch first and pip will reuse it.

For Metric3Dv2 you also need the `mmengine` + `mmcv-lite` imports the
upstream hub repo expects, and `xformers` must be absent (or exactly
ABI-matched — the prebuilt wheels on pip break on non-matching torch):

```bash
VIRTUAL_ENV=.venv uv pip install mmengine mmcv-lite
VIRTUAL_ENV=.venv uv pip uninstall xformers
```

For VGGT + MASt3R install notes see [GPU_RUNBOOK.md](./GPU_RUNBOOK.md).

## Quickstart

```bash
plumbline list-models
plumbline list-datasets

# Match a paper number in under 2 minutes on a 3090:
export NYUV2_ROOT=/path/to/nyuv2   # contains nyu_depth_v2_labeled.mat
plumbline reproduce da-v2-small-nyuv2

# View-count scaling on ETH3D courtyard (multi-view):
export ETH3D_ROOT=/path/to/eth3d
python reproductions/vggt_view_sweep_courtyard.py
```

`plumbline run` also accepts arbitrary dataset kwargs, e.g.

```bash
plumbline run --model vggt --dataset eth3d --tasks pose \
  --dataset-kwargs views_per_sample=4 --max-views 4
```

## Reproducing paper numbers

Status summary as of the latest push — see
[REPRODUCTIONS.md](./REPRODUCTIONS.md) for the authoritative table:

| Reproduction | Paper | Observed | Status |
|---|---|---|---|
| `da-v2-small-nyuv2` | AbsRel 0.053 | **0.0510** | ✅ |
| `da-v2-base-nyuv2` | AbsRel 0.049 | **0.0456** | ✅ |
| `da-v2-large-nyuv2` | AbsRel 0.045 | **0.0428** | ✅ |
| `metric3d-v2-nyuv2` | AbsRel 0.063 | **0.0660** | ✅ |
| `metric3d-v2-giant-nyuv2` | AbsRel 0.067 | **0.0702** | ✅ |
| `da3-nyuv2` | δ₁ 0.974 | **0.9684** | ✅ |

Three pipeline details separate plumbline's default from typical
"just-ran-the-HF-model" numbers and are required to hit paper targets:

1. **NYU `rawDepths`**, not the colorization-filled `depths`. Every
   modern mono-depth paper citing "NYU Eigen" evaluates against the
   sparse Kinect measurements; plumbline's loader defaults to that.
2. **Post-alignment depth clipping** (`depth_clip: [0.001, 10.0]` for
   NYU) — without this, one pathological sample can push ViT-L's mean
   AbsRel to 77 via an alignment-induced outlier.
3. **Scale+shift in inverse-depth space** (MiDaS protocol) for
   relative-depth models.

See the "Note on the NYUv2 Eigen 2014 protocol" section of
[REPRODUCTIONS.md](./REPRODUCTIONS.md) for the full diagnostic.

## Not yet reproducible without user-supplied credentials

- `vggt-paper-dtu-mvs` — v0.1 paper-match gate. Pins VGGT Table 2's
  chamfer=0.382 on DTU. Loader + YAML ready; data is public, no ToS —
  unblock by downloading DTU + the MVSNet preprocessing and setting
  `$DTU_ROOT`. **Replaces** the previous `vggt-paper-scannet-depth`
  placeholder, which assumed a VGGT paper table that doesn't exist.
- `depth-anything-v2-sintel` — Sintel loader works on the public RGB
  bundle, but the paper's AbsRel target needs the auth-gated depth +
  camera archive.
- `da-v2-small-kitti` / `metric3d-v2-kitti` — KITTI loader + Garg/Eigen
  crop masks landed; reproductions need the user to unpack the public
  KITTI raw + annotated-depth archives under `$KITTI_ROOT`. The pinned
  652-frame Eigen sample list ships in-repo at
  `reproductions/kitti_eigen_benchmark_652.txt`.

All of these run end-to-end the moment their data lands.

## Documentation

- [`plan.md`](./plan.md) — architecture + v0.1 spec and roadmap.
- [`GPU_RUNBOOK.md`](./GPU_RUNBOOK.md) — running on a rented GPU,
  including per-adapter install quirks.
- [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) — paper-number configs,
  observed values, tolerances, and protocol notes.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to add a model or
  dataset adapter.

## License

Apache-2.0.
