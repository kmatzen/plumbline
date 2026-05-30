# Sintel DA-V2 Table 2 — handoff (parked 2026-05-30)

Upstream eval archaeology: [`docs/DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md).
MoGe bundle @ 872×436 + `has_sharp_boundary` matches Table 3 (0.214 ✅); native
`$SINTEL_ROOT` + `sintel_dav2` does not.

Parallel to [`ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md). **Do not tune YAML**
to chase paper AbsRel — OFF-PAPER (~52% under). Narrative: `docs/DISCREPANCIES.md` **D32**.

## Done

| Item | Status |
|------|--------|
| Native Sintel staged (`$SINTEL_ROOT`) | ✅ |
| Protocol `sintel_dav2` (final, `max_depth=70`, `scale_shift`) | ✅ |
| Full harness `depth-anything-v2-sintel` (1064 frames) | ✅ MISMATCH |
| Pass probe (`final` vs `clean`) | ✅ 2026-05-30 |

## Key numbers (ViT-L, 1064 frames, `max_depth=70`, 2026-05-30)

| Track | AbsRel | Paper 0.487 | Δ |
|-------|--------|-------------|---|
| `final` pass (protocol default) | **0.2321** | 0.487 | −52.3 % |
| `clean` pass (same GT depth) | **0.2224** | 0.487 | −54.3 % |

`clean` is slightly *better* (lower AbsRel), not worse — pass choice does not explain the gap.

**MoGe bundle (Table 3, different protocol):** `da-v2-large-sintel-moge` **0.2139** vs **0.2140** ✅ MATCH.

## Ruled out

1. Missing sky mask — pre-fix run without `max_depth` was invalid (AbsRel exploded).
2. Wrong RGB pass — `clean` does not move metric toward paper.

## Artifacts

- `$PLUMBLINE_WORK/runs/da_v2_sintel_native_fix_20260530.json`
- `$PLUMBLINE_WORK/runs/sintel_pass_probe_20260530.log`
- MoGe: `da_v2_large_sintel_moge_20260530T170219Z.json`

## When you return

1. **DA-V2 / DA-V1 Sintel eval code** — paper Table 2 does not pin pass or sky handling;
   appendix B.9 mentions resolution scaling but not Sintel eval script.
2. **MonST3R `depth_metric.ipynb` Sintel cell** — per-seq LAD2 + `max_depth=70` (D27 family);
   compare aggregation vs plumbline per-frame mean.
3. **Depth Pro column** — `depth-pro-sintel` δ₁ 0.242 vs 0.400 (metric depth, different table).

## Resume

```bash
source scripts/pod-localssd-env.sh
export DAV2_ROOT="$PLUMBLINE_WORK/deps/depth-anything-v2"
uv run plumbline reproduce depth-anything-v2-sintel \
  -o "$PLUMBLINE_WORK/runs/da_v2_sintel_native_fix_20260530.json"
uv run python scripts/probe-sintel-pass.py
```

## Queue

- `depth-anything-v2-sintel` → **pending / OFF-PAPER** (D32)
- `depth-pro-sintel` → separate Depth Pro Table 1 experiment
