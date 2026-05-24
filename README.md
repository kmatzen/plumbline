# plumbline

A reproducible evaluation harness for 3D geometric foundation models —
think `lm-evaluation-harness`, but for models like **VGGT**, **Depth
Anything 3**, **MASt3R**, **Metric3Dv2**, and **Depth Anything V2**.

**Status:** v0.1 in development. **16 paper-match cells** across NYU +
KITTI + DIODE mono-depth, each verified against the source PDF. Pose
+ multi-view-chamfer infra landed but not yet GPU-validated against
paper cells. API will still change before 1.0.

## What works today

- **13 model adapters**: DA-V2 (6 variants including metric
  Indoor/Outdoor), DA3, Metric3Dv2 (S/L/Giant2), MoGe-1, MoGe-2 (incl.
  `*-normal`), Marigold v1-1, GeoWizard, Depth Pro, MASt3R (N-view via
  PointCloudOptimizer for N≥3, PairViewer for N=2), VGGT, π³, **CUT3R**
  (recurrent — video + unordered image collections), and **MonST3R**
  (dynamic-scene video; base dust3r inference, flow refinement scoped as
  a follow-up). The last three landed with conversion unit tests; GPU
  validation pending.
- **12 datasets**: NYUv2 (Eigen 2014, rawDepths), KITTI (Eigen 652,
  annotated GT, Garg crop), DIODE (FoV-warp loader, MoGe-paper protocol),
  ETH3D high-res multi-view, DTU MVS (22-scan test split), CO3Dv2
  (VGGT-canonical pose-eval recipe), 7-Scenes, GSO, iBims-1, Sintel
  (RGB + flow; depth gated), ScanNet (gated), **Bonn RGB-D Dynamic**
  (video depth, one-sample-per-sequence; closes the runnable-video gap).
- **16 paper-match reproductions** with `source_confidence: verified_pdf`
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

**Verified paper-match (16 cells, safe to cite):** DA-V2 (S/B/L on
NYU, KITTI; L on DIODE, KITTI-MoGe), Metric3Dv2 (L/Giant on NYU,
KITTI), MoGe-1 ViT-L (NYU, KITTI, DIODE), Marigold v1-1 (NYU), DA3
(NYU).

**Upstream-blocked (adapter+protocol audited; gap is in the released
checkpoint or a paper-private eval config — do not promote):**
GeoWizard (NYU, KITTI), Marigold (KITTI), VGGT (DTU). See
`docs/DISCREPANCIES.md` D3 / D9 / D17 / D18 / D22.

**Infra landed, GPU validation pending:** CO3Dv2 pose for VGGT
(Table 1, AUC@30 = 0.882) and MASt3R (Table 3, mAA(30) = 0.818) — both
paper targets confirmed by direct PDF read (the MASt3R cell was
re-verified 2026-05-23, closing D23). Queued as the top two GPU jobs
(`plumbline queue`). VGGT-ETH3D per-view-masked path lands 9.4 % under
paper on a 3-scene subset; full 13-scene comparison pending (D10).

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
authoritative 16-cell matrix:

| Reproduction | Paper | Observed | Status |
|---|---|---|---|
| `da-v2-small-nyuv2` | AbsRel 0.053 | **0.0510** | ✅ |
| `da-v2-large-nyuv2` | AbsRel 0.045 | **0.0428** | ✅ |
| `metric3d-v2-giant-nyuv2` | AbsRel 0.067 | **0.0702** | ✅ |
| `da3-nyuv2` | δ₁ 0.974 | **0.9684** | ✅ |
| `da-v2-small-kitti` | AbsRel 0.078 | **0.0770** | ✅ |
| `metric3d-v2-kitti` | AbsRel 0.052 | **0.0495** | ✅ |
| `marigold-v1-1-nyuv2` | AbsRel 0.055 | **0.0577** | ✅ |
| `moge-vitl-diode-both` | AbsRel 0.0400 | **0.0407** | ✅ |
| `da-v2-large-kitti-moge` | AbsRel 0.0561 | **0.0569** | ✅ |

## Not yet reproducible without user-supplied data or compute

- **CO3Dv2 pose** (`vggt-co3dv2-pose`, `mast3r-co3dv2-pose`) — infra
  + verified paper targets landed 2026-04-27; awaiting GPU run on
  CO3Dv2 (~30-50 GB stage of the 41 SEEN categories' test split).
  Will gate the pose half of v0.1.
- `vggt-paper-dtu-mvs` — protocol port complete (per-view-masked,
  PatchmatchNet filter); residual ~2 × gap is upstream-blocked. Run
  to confirm the structural correctness, not as a paper-match. Set
  `$DTU_ROOT` and run.
- `depth-anything-v2-sintel` — Sintel loader works on the public RGB
  bundle; the paper's AbsRel target needs the auth-gated depth +
  camera archive (deprioritized 2026-04-19; substitutes are GSO /
  iBims-1).

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
