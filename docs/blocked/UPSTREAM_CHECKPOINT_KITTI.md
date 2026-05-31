# BLOCKED — GeoWizard / Marigold · KITTI (plumbline cells)

| Field | Value |
|-------|--------|
| **Status** | 🔒 Fundamentally blocked **in plumbline** (explained off-paper) |
| **Repros** | `geowizard-kitti`, `marigold-v1-1-kitti`, etc. |
| **Discrepancies** | D9, D17, D18, D22 |

## Summary

These models **can** hit paper numbers on their **native** pipelines and prepared
sets. Plumbline cells stay **blocked** because the gap is **not** a fixable loader bug:

| Model | Plumbline | Paper | Root cause |
|-------|-----------|-------|------------|
| GeoWizard NYU/KITTI | ~0.057 / ~0.11 | 0.052 / 0.097 | **Best-of-N seeds** (author confirmed on GitHub #36) |
| Marigold KITTI | ~0.109 | 0.099 | **v1-1 / 1-step** default vs paper **v1-0 / 50-step** (D9) |

Single-seed plumbline eval **matches independent reproducers** (~0.0574); it will not
reach paper 0.052 without the paper-private seed cherry-pick.

## Why blocked in plumbline

Implementing best-of-N seed search would **not** be an honest single-run reproduction.
Switching Marigold to v1-0/50-step is documented as a **checkpoint-generation delta**,
not a plumbline bug — native Marigold pipeline already reproduces 0.099.

## What would unblock (policy choice)

- Document cells as **informational** only (current stance), **or**
- Add explicit `best_of_n_seeds` reproduction mode **labeled non-paper** (not done).

## Do not

- Treat off-paper plumbline numbers as adapter failures
- Tune YAML targets to absorb cherry-picked paper tables

## Links

- [`../DISCREPANCIES.md`](../DISCREPANCIES.md) D9, D17, D18, D22
- [`../BLOCKED.md`](../BLOCKED.md)
