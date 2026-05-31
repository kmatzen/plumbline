# BLOCKED — Depth Pro Table 1 · Middlebury (δ₁)

| Field | Value |
|-------|--------|
| **Status** | 🔒 Fundamentally blocked (off-paper) |
| **Repro** | `depth-pro-middlebury` |
| **Protocol** | `middlebury_depth_pro_metric` |
| **Paper** | δ₁ **0.605** (Table 1) |
| **Observed** | δ₁ **0.7589** (2026-05-31, 15/15 scenes) |
| **Direction** | Reads **better** than paper (+25.5 %) |

## Summary

Full **MiddEval3** training split at **F** resolution (~2872×1984), mask **mask0nocc**,
depth clip **0.001–10 m** per appendix Table 16 (**15** samples). Metric comparison,
no alignment. Model scores **higher** δ₁ than the paper cell.

## What we tried

| Item | Detail |
|------|--------|
| Data | `$MIDDLEBURY_ROOT` via `scripts/download-middlebury.sh` |
| Scenes | All 15 trainingF scenes completed |
| Protocol | Table 16 depth range; `middlebury_depth_pro_metric` |

No remaining plumbline knob is cited in the paper for this column.

## Why this is blocked

1. **No public Middlebury eval** in `ml-depth-pro`.
2. **Reads better** — typical of wrong mask variant, resolution handling, or **paper weights ≠ public `depth_pro.pt`**, not a simple loader bug (Booster matched on same checkpoint).
3. Middlebury δ₁ is sensitive to **which mask** (occ vs all) and **resolution**; paper does not ship a reference implementation to diff.

## What would unblock

- Author eval script or mask/resolution specification for Table 1 Middlebury, **or**
- Paper-matched weights on MiddEval3 F + mask0nocc.

## Do not

- Tune `paper_reference.value` in `depth_pro_middlebury.yaml`

## Artifacts

| Artifact | Path |
|----------|------|
| JSON | `$PLUMBLINE_WORK/runs/depth_pro_middlebury_20260531.json` |
| S3 | `s3://plumbline-bench/runs/tier_depth_pro_table1_20260531/results/` |

## Links

- [`../BLOCKED.md`](../BLOCKED.md)
