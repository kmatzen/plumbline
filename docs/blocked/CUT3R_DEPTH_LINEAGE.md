# BLOCKED — CUT3R · NYU / KITTI / Bonn depth (plumbline cells)

| Field | Value |
|-------|--------|
| **Status** | 🔒 Protocol-blocked (explained; paper confirmed elsewhere) |
| **Repros** | `cut3r-nyuv2`, `cut3r-kitti`, `cut3r-bonn`, … |
| **Discrepancy** | D24 |

## Summary

Plumbline strict depth protocol scores **better** than published CUT3R table cells
(e.g. NYU 0.052 vs paper 0.086). Re-scoring **the same cached predictions** with
CUT3R-lineage crops, clips, and GT fields still does not land on paper numbers.

**CUT3R's own pipeline** on its exact prepared sets **does** reproduce all three paper
cells within ~2 %:

| Dataset | Native CUT3R | Paper |
|---------|--------------|-------|
| NYU | 0.08595 | 0.086 |
| KITTI | 0.09219 | 0.092 |
| Bonn | 0.07661 | 0.078 |

## Why plumbline cells stay blocked

The mismatch is a **documented protocol delta** (sparse vs filled depth, eval crop,
aggregation), not wrong weights. Closing plumbline cells without adopting the full
DUSt3R-lineage eval would misrepresent what the harness measures.

## What would unblock

- Explicit `cut3r_native_eval` reproduction mode mirroring upstream scripts (informational), **or**
- Accept `paper_match: no` with D24 explanation (current).

## Do not

- Tune `paper_reference` to plumbline's stricter numbers and claim ✅

## Links

- [`../DISCREPANCIES.md`](../DISCREPANCIES.md) D24
- [`../BLOCKED.md`](../BLOCKED.md)
