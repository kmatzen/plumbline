# plumbline

A reproducible evaluation harness for 3D geometric foundation models —
think `lm-evaluation-harness`, but for models like **VGGT**, **Depth
Anything 3**, **MASt3R**, **Metric3Dv2**, and **Depth Anything V2**.

> **License note:** plumbline's own code is Apache-2.0, but the package
> **bundles** vendored upstream model code (DAGE / CUT3R / DUSt3R / MASt3R /
> MonST3R) under **NonCommercial** licenses (CC BY-NC[-SA]). The distribution as
> a whole is therefore usable for **non-commercial purposes only**. See
> [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

**Status:** v0.2 in development. **47 paper-match cells** — 39 mono-depth
(NYU / KITTI / DIODE / GSO / iBims-1 / ETH3D MoGe-eval / DDAD / Sintel MoGe /
Depth Pro Booster + Sun-RGBD / UniK3D metric NYU / CUT3R NYU·KITTI·Bonn) + 7 multi-view pose/trajectory
(CO3Dv2 / Sintel / TUM-Dynamics) — each verified against the source PDF.
API will still change before 1.0.

## What works today

- **12 model adapters**: DA-V2 (6 variants including metric
  Indoor/Outdoor), DA3, Metric3Dv2 (S/L/Giant2), MoGe-1, MoGe-2 (incl.
  `*-normal`), Marigold v1-1, GeoWizard, Depth Pro, MASt3R (N-view via
  PointCloudOptimizer for N≥3, PairViewer for N=2), VGGT, **CUT3R**
  (recurrent — video + unordered image collections), and **MonST3R**
  (dynamic-scene video; the v1.2 `video_pose` path wires MonST3R's full
  flow + motion-mask + temporal global alignment, GPU-validated on the
  Sintel Table-4 trajectory cell). CUT3R also has a GPU smoke run
  (informational).
- **12 datasets**: NYUv2 (Eigen 2014, rawDepths), KITTI (Eigen 652,
  annotated GT, Garg crop), DIODE (FoV-warp loader, MoGe-paper protocol),
  ETH3D high-res multi-view, DTU MVS (22-scan test split), CO3Dv2
  (VGGT-canonical pose-eval recipe), 7-Scenes, GSO, iBims-1, Sintel
  (RGB + flow; depth gated), Bonn RGB-D Dynamic
  (video depth, one-sample-per-sequence), **TUM-Dynamics** (freiburg3
  video-pose, MonST3R/DAGE Table 4 trajectory eval).
- **47 paper-match reproductions** with `source_confidence: verified_pdf`
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

**Verified paper-match (36 cells, safe to cite):** 30 mono-depth — DA-V2
(S/L on NYU; S/B/L on KITTI; L on DIODE + KITTI-MoGe + GSO + ETH3D
MoGe-eval + iBims-1 + DDAD + Sintel MoGe), Metric3Dv2 (L/Giant on NYU + KITTI), MoGe-1 ViT-L (NYU,
KITTI, DIODE, GSO, iBims-1, ETH3D MoGe-eval, DDAD, Sintel MoGe), Marigold v1-1 (NYU), DA3
(NYU δ₁), MonST3R (NYU), DUSt3R (KITTI), Depth Pro (Booster + Sun-RGBD Table 1, δ₁);
plus 6 multi-view pose — VGGT / MASt3R / DUSt3R on CO3Dv2 (mAA@30),
MonST3R on Sintel (trajectory ATE, Table 4), and DAGE on Sintel + TUM-Dynamics
(trajectory ATE, Table 4).

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

The DAGE and dust3r-lineage (CUT3R / DUSt3R / MASt3R / MonST3R) model
code is **vendored** in the wheel — no clones. Each adapter needs only a
few runtime pip deps, installed via `plumbline install <model>` (e.g.
`plumbline install dust3r` → `roma scikit-learn trimesh`); `plumbline
doctor` reports what's missing. CUT3R additionally builds the vendored
`curope` CUDA ext and needs its 512-DPT checkpoint. `install.py` is the
single source of truth for every adapter's dependencies.

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
authoritative 47-cell matrix:

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
| `dage-tum-pose` | ATE 0.014 | **0.0136** | ✅ |

## Not yet reproducible without user-supplied data or compute

- `vggt-paper-dtu-mvs` — protocol port complete (per-view-masked,
  PatchmatchNet filter); residual ~2 × gap is upstream-blocked. Run
  to confirm the structural correctness, not as a paper-match. Set
  `$DTU_ROOT` and run.
- **DA-V2 Table 2 native ETH3D / Sintel** — staged and run end-to-end; **parked
  OFF-PAPER** (~30–52 % *under* paper; MoGe-bundle cells on same datasets still
  ✅). Return notes: [`docs/ETH3D_DAV2_TABLE2_HANDOFF.md`](docs/ETH3D_DAV2_TABLE2_HANDOFF.md),
  [`docs/SINTEL_DAV2_TABLE2_HANDOFF.md`](docs/SINTEL_DAV2_TABLE2_HANDOFF.md).
- `depth-pro-sintel` — Depth Pro δ₁ **0.242** vs **0.400** (metric depth; separate
  from DA-V2 native Sintel above).

## Extend it without cloning

Add your own model or dataset from a **separate** pip package — no fork required.
`pip install plumbline-bench`, register an adapter, and advertise it via the
`plumbline.adapters` entry-point group:

```python
# my_package/adapters.py
from plumbline import Model, ModelCapabilities, Prediction, register_model

@register_model("my-model")
class MyAdapter(Model):
    capabilities = ModelCapabilities(tasks=frozenset({"mono_depth"}), is_metric=True)
    def predict(self, images, intrinsics=None) -> Prediction: ...
```

```toml
# my_package/pyproject.toml
[project.entry-points."plumbline.adapters"]
my_adapters = "my_package.adapters"
```

plumbline auto-discovers it — `plumbline list-models` and `plumbline run my-model …`
just work. See [`CONTRIBUTING.md`](./CONTRIBUTING.md#extending-plumbline-without-cloning-it-plugins).

## Documentation

- [`docs/README.md`](./docs/README.md) — map of all docs (start here)
- [`GPU_RUNBOOK.md`](./GPU_RUNBOOK.md) — GPU bring-up, queue, active work
- [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) — paper-match matrix
- [`docs/DISCREPANCIES.md`](./docs/DISCREPANCIES.md) — open issues (D-numbers)
- [`docs/CONFIDENCE_AUDIT.md`](./docs/CONFIDENCE_AUDIT.md) — where each off-paper gap lives, confirmed vs unknown
- [`docs/BLOCKED.md`](./docs/BLOCKED.md) — fundamentally blocked cells
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — extend the harness
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — dev setup and PR bar
- [`docs/SOURCE_AUDIT.md`](./docs/SOURCE_AUDIT.md) — adapter vs upstream source

## License

plumbline's own source code is **Apache-2.0** (see [LICENSE](./LICENSE)).

The distributed package additionally **bundles** vendored upstream model code
under `src/plumbline/_vendor/` — DAGE (CC BY-NC 4.0) and CUT3R / DUSt3R / MASt3R
/ MonST3R (CC BY-NC-SA 4.0). Because these are **NonCommercial** licenses, the
package as a whole may be used for **non-commercial purposes only**. Each
vendored tree keeps its own `LICENSE`; the full inventory is in
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md). Models whose licenses do not
permit redistribution (GeoWizard, and bespoke-license VGGT / Depth Pro) are
**not** vendored — they install from their upstream source.
