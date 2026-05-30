# DA-V2 Table 2 — upstream eval archaeology (2026-05-30)

Parked native ETH3D / Sintel / DIODE cells share one pattern: **plumbline
devkit loaders + `scale_shift`** read *better* than paper. This doc records
where the authors' numbers likely come from and what to run instead.

## Executive summary

| Dataset | DA-V2 repo ships eval? | Closest public recipe | plumbline native | plumbline MoGe bundle |
|---------|------------------------|----------------------|------------------|----------------------|
| ETH3D | **No** | MoGe `eval_baseline.py` + HF bundle @ **2048×1365** | OFF-PAPER (−32 %) | ✅ Table 3 @ 0.047 |
| Sintel | **No** | MoGe bundle @ **872×436**, `has_sharp_boundary` | OFF-PAPER (−52 %) | ✅ Table 3 @ 0.214 |
| DIODE | **No** | MoGe bundle @ **1024×768** + disparity clamp | OFF-PAPER (+200 %) | ~0.062 Table-2 align (still under 0.073) |

**Authors' own words** (arXiv:2406.09414, Table 2 caption): metrics on legacy
benchmarks *"cannot be correctly reflected"* for V2's strengths; they built
**DA-2K** for relative-depth evaluation instead.

**Community state:** [Issue #280](https://github.com/DepthAnything/Depth-Anything-V2/issues/280) (DIODE),
[#281](https://github.com/DepthAnything/Depth-Anything-V2/issues/281) (ETH3D) —
no official zero-shot eval script; reproducers get ~0.2 DIODE / ~0.5 ETH3D with
devkit GT + `lstsq` alignment (matches our native path, not paper).

## The MoGe eval harness (best public match)

MoGe ships the **de-facto cross-paper** benchmark pipeline for DA-V2 Table 2:

- Data: [Ruicheng/monocular-geometry-evaluation](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) (same trees as `$*_MOGE_ROOT`)
- Loader: `moge.test.dataloader.EvalDataLoaderPipeline` (homographic FoV warp)
- DA-V2 wrapper: [`moge/baselines/da_v2.py`](https://github.com/microsoft/MoGe/blob/main/baselines/da_v2.py)
- Metrics: `moge.test.metrics.compute_metrics` on **`disparity_affine_invariant`**
  with `align_affine_lstsq` + **`pred_disp.clamp_min(1 / gt_depth[mask].max())`**
  — equivalent to plumbline **`scale_shift_clamped`**, not plain `scale_shift`.

Docs: [`moge/docs/eval.md`](https://github.com/microsoft/MoGe/blob/main/docs/eval.md)

```bash
# From a MoGe clone, with data/eval/DIODE/ unzipped from HF:
python moge/scripts/eval_baseline.py \
  --baseline baselines/da_v2.py \
  --config configs/eval/benchmarks/diode.json \
  --output eval_output/da_v2_vitl_diode.json \
  --repo /path/to/depth-anything-v2 --backbone vitl
```

Benchmark geometry (from `configs/eval/benchmarks/`):

| Benchmark | Warp size | Notes |
|-----------|-----------|-------|
| ETH3D | 2048 × 1365 | `include_segmentation: true`, `depth_unit: 1` |
| DIODE | 1024 × 768 | `include_segmentation: true` |
| Sintel | 872 × 436 | `has_sharp_boundary: true` |

Plumbline ports this warp in `DIODEMogeEvalLoader`, `ETH3DMogeEvalDataset`, etc.

## Per-dataset handoffs (native path)

| D# | Dataset | Handoff |
|----|---------|---------|
| D31/D33 | ETH3D | [`ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md) — loader RGB/GT fix ✅; z-buffer @518 still ~−32 % under paper |
| D32 | Sintel | [`SINTEL_DAV2_TABLE2_HANDOFF.md`](SINTEL_DAV2_TABLE2_HANDOFF.md) — `final`/`clean` pass does not explain gap |
| D29 | DIODE | [`D29_DIODE_TABLE2_HANDOFF.md`](D29_DIODE_TABLE2_HANDOFF.md) — outdoor native broken; MoGe warp no-op at native res; bundle + `scale_shift` → 0.062 vs 0.073 |

## DA-V2 inference details (MoGe baseline vs plumbline)

MoGe `baselines/da_v2.py`:

- Short-side **518**, bicubic resize, dims rounded to **multiple of 14**
- `model(image)` → **disparity** (not `infer_image`)
- ImageNet normalize

Plumbline `DepthAnythingV2Adapter` uses `infer_image` at `input_size=518` — close
but verify forward path matches (disparity vs depth) on any protocol audit.

## Depth Anything V1

V1 README reports the **same Table-2-style numbers** (ETH3D 0.127, DIODE 0.066, …)
but V1 repo also has **no** zero-shot ETH3D/Sintel/DIODE eval script — only
`metric_depth/` for NYUv2/KITTI metric models.

## DepthAnythingAC

[`evaluate_depth.py`](https://github.com/HVision-NKU/DepthAnythingAC/blob/master/evaluate_depth.py)
lists `ETH3D`, `Sintel`, `DIODE` — uses `Disparity2Depth` + `depth_cap=80`.
Suggested on issue #280; **not** verified to reproduce DA-V2 Table 2 here.

## Recommended plumbline stance

1. **Do not tune** native `diode_dav2` / `eth3d_dav2` / `sintel_dav2` YAMLs to chase Table 2.
2. Treat **MoGe-bundle Table 3 cells** as the verified cross-paper metric for those datasets.
3. For Table-2 *paper-number* audit, run or cite **MoGe `eval_baseline.py`** on HF bundles.
4. Optional experiment: add informational rows `da-v2-*-{eth3d,sintel,diode}-moge-bundle`
   with `scale_shift_clamped` (MoGe metric parity) — distinct from native blocked cells.

## Clone paths on pod

```text
$PLUMBLINE_WORK/deps/depth-anything-v2   # DA-V2
$PLUMBLINE_WORK/deps/depth-anything-v1   # V1 (no Table-2 eval)
$PLUMBLINE_WORK/deps/moge                # eval harness + da_v2 baseline
```

## MoGe harness caveat (2026-05-30)

Running `eval_baseline.py` from `$PLUMBLINE_WORK/deps/moge` on this pod hit
`NameError: read_meta` in `moge/test/dataloader.py` line 96 — upstream should
use `read_json` (as in `moge/train/dataloader.py`). Also avoid naming conflict:
`import pipeline` (PyPI `pipeline` package) must not shadow MoGe's pipeline
when launched from plumbline's venv. **Workaround:** patch `read_meta` →
`read_json` in the MoGe clone, or install MoGe in an isolated env per
`moge/docs/eval.md`.

Until fixed, plumbline MoGe-bundle repros are the verified cross-check
(`da-v2-large-diode` Table 3 ✅ 0.053 with `scale_shift_clamped`).

## Resume commands

```bash
source /mnt/localssd/plumbline/scripts/pod-localssd-env.sh
export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"

# Plumbline MoGe-bundle (already MATCH for Table 3)
uv run plumbline reproduce da-v2-large-diode  # diode_moge protocol

# MoGe upstream harness (DIODE only config)
cd "$PLUMBLINE_WORK/deps/moge"
ln -sfn "$DIODE_MOGE_ROOT/DIODE" data/eval/DIODE  # once per layout
uv run python moge/scripts/eval_baseline.py \
  --baseline baselines/da_v2.py \
  --config configs/eval/benchmarks/diode.json \
  -o "$PLUMBLINE_WORK/runs/moge_upstream_da_v2_diode_vitl.json" \
  --repo "$DAV2_ROOT" --backbone vitl
```
