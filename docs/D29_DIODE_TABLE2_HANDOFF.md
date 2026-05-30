# D29 — DA-V2 Table 2 native DIODE handoff (2026-05-30)

**Status:** Investigated; native cells stay **blocked**. Outdoor gap **explained**;
full Table-2 match not closed (bundle path reads *under* paper).

## Numbers

| Path | ViT-S | ViT-B | ViT-L | Paper S/B/L |
|------|-------|-------|-------|-------------|
| `diode_dav2` native | 0.2196 | 0.2182 | 0.2142 | 0.073 / 0.068 / 0.066 |
| `diode_dav2_moge_bundle` + `scale_shift` | **0.0618** | **0.0588** | **0.0543** | same |
| `diode_dav2_moge_bundle` + `scale_shift_clamped` | **0.0585** | — | — | same |
| MoGe Table 3 (`diode_moge` + clamped) | — | — | 0.0529 ✅ | 0.053 |

### Per-domain (`diode_dav2_moge_bundle`, scale_shift)

| Domain | ViT-S | ViT-L | n |
|--------|-------|-------|---|
| indoor | 0.0519 | 0.0406 | 325 |
| outdoor | 0.0690 | 0.0643 | 446 |

Native outdoor alone was ~0.327 (ViT-S); bundle brings outdoor to ~0.07.

## Ruled in / ruled out

| Hypothesis | Result |
|------------|--------|
| Model / weights wrong | ❌ Indoor native ≈ paper |
| `[1e-3, 50]` clip | ❌ Widening hurts outdoor |
| Homographic FoV warp on native 1024×768 | ❌ **No-op** (identical pixels) |
| MoGe HF bundle depth + `isfinite` mask | ✅ Fixes outdoor; combined still ~15–18 % **under** paper |

GT probe (`scripts/probe-diode-d29-native-vs-bundle.py`, 40 pairs): valid fraction
native **0.987** vs bundle **0.872**; depth MAE on overlap **0.0001 m**.

## Artifacts

**Local** (`$PLUMBLINE_WORK/runs/`):

- `da_v2_small_diode_moge_bundle_20260530.json`
- `da_v2_base_diode_moge_bundle_*.json`
- `da_v2_large_diode_moge_bundle_20260530.json`
- `da_v2_small_diode_moge_bundle_clamped_*.json`
- `diode_d29_warp_probe_outdoor40.log`

**S3:** `s3://plumbline-bench/runs/tier_d29_diode_20260530/results/`

**Env:** `$DIODE_MOGE_ROOT` → `$PLUMBLINE_WORK/data/diode_moge` (see `pod-localssd-env.sh`).

## When you return

1. Confirm with DA-V2 authors whether Table-2 DIODE used MoGe-lineage GT/mask or raw devkit.
2. If bundle is correct: decide whether to document **protocol delta** (native vs bundle) vs
   add informational `diode_dav2_moge_bundle` cells — still MISMATCH (under paper).
3. Do **not** repoint `diode_dav2` YAML without upstream confirmation (GPU_RUNBOOK).

## Commands

```bash
source scripts/pod-localssd-env.sh
export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"

uv run plumbline reproduce da-v2-small-diode-moge-bundle \
  -o "$PLUMBLINE_WORK/runs/da_v2_small_diode_moge_bundle.json"

uv run python scripts/probe-diode-d29-warp.py --domain outdoor --max-samples 50
uv run python scripts/probe-diode-d29-native-vs-bundle.py --max-pairs 40
```
