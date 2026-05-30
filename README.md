# plumbline

A reproducible evaluation harness for 3D geometric foundation models —
think `lm-evaluation-harness`, but for models like **VGGT**, **Depth
Anything 3**, **MASt3R**, **Metric3Dv2**, and **Depth Anything V2**.

**Status:** v0.1 in development. **27 paper-match cells** — 23 mono-depth
(NYU / KITTI / DIODE / GSO / iBims-1 / ETH3D MoGe-eval) + 4 multi-view
pose/trajectory (CO3Dv2 / Sintel) — each verified against the source PDF.
API will still change before 1.0.

## What works today

- **13 model adapters**: DA-V2 (6 variants including metric
  Indoor/Outdoor), DA3, Metric3Dv2 (S/L/Giant2), MoGe-1, MoGe-2 (incl.
  `*-normal`), Marigold v1-1, GeoWizard, Depth Pro, MASt3R (N-view via
  PointCloudOptimizer for N≥3, PairViewer for N=2), VGGT, π³, **CUT3R**
  (recurrent — video + unordered image collections), and **MonST3R**
  (dynamic-scene video; the v1.2 `video_pose` path wires MonST3R's full
  flow + motion-mask + temporal global alignment, GPU-validated on the
  Sintel Table-4 trajectory cell). π³ and CUT3R also have GPU smoke runs
  (informational).
- **12 datasets**: NYUv2 (Eigen 2014, rawDepths), KITTI (Eigen 652,
  annotated GT, Garg crop), DIODE (FoV-warp loader, MoGe-paper protocol),
  ETH3D high-res multi-view, DTU MVS (22-scan test split), CO3Dv2
  (VGGT-canonical pose-eval recipe), 7-Scenes, GSO, iBims-1, Sintel
  (RGB + flow; depth gated), ScanNet (gated), **Bonn RGB-D Dynamic**
  (video depth, one-sample-per-sequence; closes the runnable-video gap).
- **27 paper-match reproductions** with `source_confidence: verified_pdf`
  — see [REPRODUCTIONS.md](./REPRODUCTIONS.md). Each cell audited
  table-+-column-+-row against the source paper
  ([reproductions/AUDIT.md](./reproductions/AUDIT.md)).
- **Pose metrics** include both absolute per-view and **pairwise
  relative-pose AUC** (the aggregation VGGT / MASt3R / DUSt3R papers
  report), plus VGGT/PoseDiffusion-canonical extensions: RRA/RTA at τ,
  antipodal translation, and the 1°-bin histogram AUC mode the paper
  cells use.
- **7-DoF similarity alignment** (Umeyama via camera-centre
  correspondences) wired through the runner for ETH3D / DTU / T&T
  chamfer. Per-view-masked chamfer (CUT3R / MASt3R lineage) lands the
  structurally-correct DTU and ETH3D protocols.

## What's verified vs upstream-blocked

Plumbline distinguishes between "we built honest infra and the paper
cell reproduces" and "we built honest infra but the public release
doesn't reproduce the paper cell". The matrix in
[`REPRODUCTIONS.md`](./REPRODUCTIONS.md) tracks this:

**Verified paper-match (27 cells, safe to cite):** 23 mono-depth — DA-V2
(S/L on NYU; S/B/L on KITTI; L on DIODE + KITTI-MoGe + GSO + ETH3D
MoGe-eval), Metric3Dv2 (L/Giant on NYU + KITTI), MoGe-1 ViT-L (NYU,
KITTI, DIODE, GSO, iBims-1, ETH3D MoGe-eval), Marigold v1-1 (NYU), DA3
(NYU δ₁), MonST3R (NYU), DUSt3R (KITTI); plus 4 multi-view pose — VGGT
/ MASt3R / DUSt3R on CO3Dv2 (mAA@30) and MonST3R on Sintel (trajectory
ATE, Table 4).

**Upstream-blocked (adapter+protocol audited; gap is in the released
checkpoint or a paper-private eval config — do not promote):**
GeoWizard (NYU, KITTI), Marigold (KITTI), VGGT (DTU). See
`docs/DISCREPANCIES.md` D3 / D9 / D17 / D18 / D22.

**Off-paper, investigated (honest infra; documented protocol / recipe
deltas, not promoted to ✅):** VGGT-ETH3D 13-scene Overall 0.875 vs 0.709
(+23.5 %, driven by the `terrains` outlier — D10); DUSt3R / MonST3R /
CUT3R lineage-depth cells on NYU / Bonn / Sintel (GT-processing recipe
deltas — D24 / D27 / D28). See `docs/DISCREPANCIES.md`.

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
plumbline queue            # the GPU backlog: what to run, footprints, env vars

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

A handful of representative ✅ reproductions across the three
datasets — see [REPRODUCTIONS.md](./REPRODUCTIONS.md) for the
authoritative 27-cell matrix:

| Reproduction | Paper | Observed | Status |
|---|---|---|---|
| `da-v2-small-nyuv2` | AbsRel 0.053 | **0.0510** | ✅ |
| `da-v2-large-nyuv2` | AbsRel 0.0420 | **0.0428** | ✅ |
| `metric3d-v2-giant-nyuv2` | AbsRel 0.067 | **0.0702** | ✅ |
| `da3-nyuv2` | δ₁ 0.974 | **0.9684** | ✅ |
| `da-v2-small-kitti` | AbsRel 0.078 | **0.0770** | ✅ |
| `metric3d-v2-kitti` | AbsRel 0.052 | **0.0495** | ✅ |
| `marigold-v1-1-nyuv2` | AbsRel 0.055 | **0.0577** | ✅ |
| `moge-vitl-diode-both` | AbsRel 0.0400 | **0.0407** | ✅ |
| `da-v2-large-kitti-moge` | AbsRel 0.0561 | **0.0569** | ✅ |
| `vggt-co3dv2-pose` | AUC@30 0.882 | **0.8964** | ✅ |
| `monst3r-sintel-pose` | ATE 0.108 | **0.1134** | ✅ |

## Not yet reproducible without user-supplied data or compute

- `vggt-paper-dtu-mvs` — protocol port complete (per-view-masked,
  PatchmatchNet filter); residual ~2 × gap is upstream-blocked. Run
  to confirm the structural correctness, not as a paper-match. Set
  `$DTU_ROOT` and run.
- `depth-anything-v2-sintel` — Sintel loader works on the public RGB
  bundle. (Sintel depth + camera archives turned out to be direct
  downloads, not auth-gated — now staged on S3; see GPU_RUNBOOK. The
  first metric-depth Sintel run, `depth-pro-sintel`, is off-paper at
  δ₁ 0.2418 vs 0.400 — an open experiment, see DISCREPANCIES.md.)

## Documentation

- [`plan.md`](./plan.md) — architecture + v0.1 spec and roadmap.
- [`GPU_RUNBOOK.md`](./GPU_RUNBOOK.md) — running on a rented GPU,
  including per-adapter install quirks.
- [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) — paper-number configs,
  observed values, tolerances, and protocol notes.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to add a model or
  dataset adapter.
- [`docs/SOURCE_AUDIT.md`](./docs/SOURCE_AUDIT.md) — per-adapter audit of
  plumbline's implementation against each method's released upstream
  source code (preprocessing, forward, output conventions).

## License

Apache-2.0.
