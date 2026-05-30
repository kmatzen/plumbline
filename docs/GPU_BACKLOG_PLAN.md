# GPU backlog plan (2026-05-30)

Executable plan after the MoGe mono-depth coverage pass. The GPU queue has
**no `pending` jobs** — off-paper native Table-2 cells are **`blocked`**. This
doc is the single place to pick up work; [`GPU_RUNBOOK.md`](../GPU_RUNBOOK.md)
still governs thrift, S3, and the no-YAML-tune rule.

## Status snapshot

| Track | State | Doc |
|-------|--------|-----|
| MoGe-bundle DA-V2 / MoGe-1 (NYU, KITTI, DIODE, GSO, iBims, ETH3D, DDAD, Sintel) | ✅ done | `REPRODUCTIONS.md` |
| DA-V2 native ETH3D / Sintel Table 2 | 🔎 parked (upstream recipe) | `ETH3D_DAV2_TABLE2_HANDOFF.md`, `SINTEL_DAV2_TABLE2_HANDOFF.md` |
| Depth Pro Sintel Table 1 | 🔎 blocked (no public eval) | D32 notes in `DISCREPANCIES.md` |
| VGGT ETH3D 13-scene chamfer | ✅ ran (D10 MISMATCH documented) | `vggt_eth3d_multiscene_chamfer.yaml` |
| **D29 native DIODE outdoor** | 🔎 parked (bundle ~under paper) | D29, `DA_V2_TABLE2_UPSTREAM_EVAL.md` |

## Execution order

### 1. D29 — native DIODE outdoor (parked)

**Goal:** Close DA-V2 Table 2 `da-v2-{small,base,large}-diode-native` without
changing pinned paper targets.

**Evidence (baseline):**

- Indoor native: AbsRel **0.072** vs paper **0.073** (matches).
- Outdoor native: **0.327** (drives `domain=both` miss).
- MoGe Table 3 bundle (`diode_moge` + `scale_shift_clamped`): **0.053** ✅.

**2026-05-30 experiments:**

| Experiment | ViT-S AbsRel | ViT-L AbsRel | vs Table 2 |
|------------|--------------|--------------|------------|
| `diode_dav2` native | 0.2196 | 0.2142 | +200 % (blocked) |
| `diode_dav2_moge_warp` (FoV warp on native files) | 0.2196 | — | **no-op** at 1024×768 (image diff 0) |
| `diode_dav2_moge_bundle` + `scale_shift` | **0.0618** | **0.0588** | **0.0543** | −15 % / −13 % / −18 % under |
| `diode_dav2_moge_bundle` + `scale_shift_clamped` | **0.0585** | — | — | −20 % under (ViT-S) |

**Per-domain (bundle + scale_shift):** indoor ViT-S **0.052** / outdoor **0.069** (n=325/446).
Native outdoor was **~0.33**; bundle fixes outdoor but both domains sit *under* paper.

GT probe (40 pairs): native valid **98.7%** vs bundle **87.2%** mask coverage; depth MAE **0.1 mm** on overlap.

**Verdict:** Native `diode_dav2` cells stay **blocked** — outdoor needs MoGe-bundle GT/mask,
not FoV warp. Bundle + Table-2 alignment still MISMATCH (under paper, like D31/D32).
See [`docs/D29_DIODE_TABLE2_HANDOFF.md`](D29_DIODE_TABLE2_HANDOFF.md).

**Next (D29):** upstream confirmation only; no further GPU unless authors reply.

**Artifacts:** `$PLUMBLINE_WORK/runs/diode_d29_warp_probe_outdoor40.log`,
`da_v2_small_diode_moge_bundle_20260530.json`, `da_v2_large_diode_moge_bundle_20260530.json`.

**Do not:** Widen `[1e-3, 50]` clip or tune reproduction YAML paper values.

### 2. DA-V2 upstream eval archaeology — **documented** ✅

See [`docs/DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md). MoGe
`eval_baseline.py` + HF bundles is the reproducible Table-2 path; DA-V2 repo ships
no native ETH3D/Sintel/DIODE eval.

Optional GPU: run MoGe harness on DIODE/ETH3D/Sintel configs to pin exact upstream
numbers on this pod.

**Also done (archaeology):** DA-V1 has no Table-2 eval; issues [#280](https://github.com/DepthAnything/Depth-Anything-V2/issues/280) / [#281](https://github.com/DepthAnything/Depth-Anything-V2/issues/281); ETH3D frame inventory 454=454 ([handoff](ETH3D_DAV2_TABLE2_HANDOFF.md)).

### 3. VGGT ETH3D 13-scene (D10 closure)

**Status:** Full 13-scene run already recorded in `vggt_eth3d_multiscene_chamfer.yaml`
(Overall **0.875** vs **0.709**, terrains outlier). No further GPU unless re-run for
JSON/S3 provenance only.

**Optional:** Sync result JSON to S3 if missing; mark queue row as investigated.

### 4. Depth Pro Table 1 (defer)

Sintel δ₁ confirmed off-paper with weights. Next column (ETH3D δ₁ 0.415) needs a
**metric eval set** definition before GPU — not the chamfer/z-buffer path used for DA-V2.

### 5. Defer (high friction)

- RealEstate10K pose ×3 (~20 GB YouTube scrape).
- ScanNet (ToS).
- VGGT-DTU / GeoWizard / Marigold-KITTI re-runs (upstream-blocked).

## Do not spend GPU on

- Native DA-V2 ETH3D/Sintel Table 2 re-runs (parked; harness OK).
- `depth-pro-sintel` (blocked; no upstream Sintel eval).
- `plumbline queue --run` without new `pending` rows.

## Session habits

```bash
source /mnt/localssd/plumbline/scripts/pod-localssd-env.sh
./scripts/backup-session.sh <tag>   # after successful runs
git push origin main                # after queue/matrix updates
```

## Links

- Queue: `reproductions/gpu_queue.yaml`
- Matrix: `REPRODUCTIONS.md`
- Issues: `docs/DISCREPANCIES.md` (D29–D33)
