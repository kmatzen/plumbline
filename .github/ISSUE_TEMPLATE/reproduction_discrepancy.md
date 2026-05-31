---
name: Reproduction discrepancy
about: A paper number doesn't reproduce, or a ✅ cell looks wrong
title: "[repro] <model> × <dataset>: observed X vs paper Y"
labels: reproduction
assignees: ""
---

## The cell

- Reproduction / YAML: `reproductions/<file>.yaml`
- Model × dataset:
- Paper reference (table + column + row):
- Paper value:
- Observed value:
- Relative delta:

## Protocol

<!-- Which protocol preset / dataset-prep settings? Eigen vs Garg crop,
     alignment (median / scale_shift / scale_shift_clamped), depth clip,
     sample list, number of frames, etc. -->

## What you ran

```bash
plumbline reproduce <name>
# or
plumbline run --model ... --dataset ... --tasks ...
```

## Notes

<!-- Anything suggesting whether this is an adapter bug, a protocol delta, or
     a paper-private eval recipe. See docs/DISCREPANCIES.md for prior
     investigations (D-numbers). -->

## Environment

- `plumbline --version`:
- torch + CUDA, GPU:
