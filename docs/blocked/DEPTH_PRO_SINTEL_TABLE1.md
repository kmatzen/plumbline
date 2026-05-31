# BLOCKED — Depth Pro Table 1 · Sintel (δ₁)

| Field | Value |
|-------|--------|
| **Status** | 🔒 Fundamentally blocked (off-paper) |
| **Repro** | `depth-pro-sintel` |
| **Protocol** | `sintel_depth_pro_metric` (appendix Table 16) |
| **Paper** | δ₁ **0.400** (Bochkovskii et al. 2024, Table 1) |
| **Observed** | δ₁ **0.2409** (2026-05-31, 1064/1064 frames) |
| **Direction** | Reads **better** than paper (−40 % vs target) |

## Summary

Plumbline aligned the eval with Depth Pro appendix **Table 16**: metric depth, **no**
scale alignment, valid range **0.01–80 m**, **1064** training frames, bilinear pred→GT.
The gap **persisted** after moving from legacy `sintel_dav2` (0.2418) to
`sintel_depth_pro_metric` (0.2409). Depth-range and pass levers are ruled out.

## What we tried

| Lever | Result |
|-------|--------|
| `max_depth` 70 vs 80 m | No meaningful δ₁ change on full set |
| `final` vs `clean` pass | Both under paper (DA-V2 probe: clean slightly *better*) |
| 80-frame smoke vs full 1064 | Smoke ~0.48; tail scenes pull mean to ~0.24 |
| Per-scene aggregation | Equal-scene mean ~0.23; excl. six ultra-easy scenes → ~0.31, still under 0.40 |
| iBims sanity (same weights) | δ₁ **0.8458** on 100 frames — adapter + checkpoint OK on indoor GT |

### Per-scene breakdown (2026-05-31)

From `scripts/analyze-depth-pro-sintel-json.py` on `depth_pro_sintel_table16_20260531.json`:

| Finding | Value |
|---------|--------|
| First 80 frames (loader order) | δ₁ **0.477** |
| Frames 81–1064 | δ₁ **0.221** |
| Equal-scene mean (23 scenes) | δ₁ **0.230** |
| NaN frames | 19 / 1064 |

Six scenes with scene-mean δ₁ &lt; 0.05 (300 frames): `mountain_1`, `bandage_1`,
`sleeping_2`, `bamboo_1`, `temple_2`, `bamboo_2`. Best scenes: `shaman_2` **0.76**,
`alley_1` **0.60**.

## Why this is blocked

1. **`ml-depth-pro` ships no Sintel eval** — no script, mask, or frame list to diff.
2. **Public weights** — README states reference weights were **re-trained** and may not match the paper table.
3. **Possible undisclosed aggregation** — paper may average per-sequence differently or use a subset; we cannot infer without authors.

Re-tuning plumbline YAML or clip ranges would **not** be evidence of reproduction.

## What would unblock

- Depth Pro authors publish eval code + exact frame list / masks, **or**
- A checkpoint proven to match the paper table on Sintel, **or**
- Written confirmation of aggregation (per-frame vs per-scene mean, scene filter).

## Do not

- Tune `paper_reference.value` in `depth_pro_sintel.yaml`
- Re-queue GPU hoping `pass_name` / `max_depth` alone will close the gap

## Artifacts

| Artifact | Path |
|----------|------|
| JSON | `$PLUMBLINE_WORK/runs/depth_pro_sintel_table16_20260531.json` |
| S3 | `s3://plumbline-bench/runs/tier_depth_pro_table1_20260531/results/` |
| Analysis | `scripts/analyze-depth-pro-sintel-json.py` |

## Links

- [`../BLOCKED.md`](../BLOCKED.md) · [`../README.md`](../README.md)
- `docs/DISCREPANCIES.md` (D32)
