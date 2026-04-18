# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

## Running

Set the appropriate dataset-root env var first; YAML files deliberately
don't hardcode machine-specific paths:

```bash
export NYUV2_ROOT=~/data/nyuv2      # for any da-v2-*-nyuv2 reproduction
export SCANNET_ROOT=~/data/scannet  # for vggt-paper-scannet-depth
export SINTEL_ROOT=~/data/sintel    # for depth-anything-v2-sintel

plumbline reproduce <name>
```

This loads `reproductions/<name>.yaml`, runs the model on the dataset,
computes metrics, and compares the primary metric against the published
value.

## Status matrix

| Name | Paper | Primary metric | Published | Observed | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `da-v2-small-nyuv2` | DA-V2 Small, NYU Eigen test | `abs_rel` | 0.063 | **0.0623** | ±5% | ✅ **match** (MPS, 4 min, 8 GB RAM OK) |
| `da-v2-large-nyuv2` | DA-V2 Large, NYU Eigen test | `abs_rel` | 0.043 | — | ±10% | GPU-only: 1.3 GB weights thrashed swap on 8 GB Mac |
| `vggt-paper-scannet-depth` | VGGT, ScanNet, 8 views | `abs_rel` | _TBD_ | — | ±5% | placeholder — awaiting GPU run |
| `depth-anything-v2-sintel` | DA-V2, Sintel | `abs_rel` | ≈0.075 | — | ±15% | blocked on Sintel depth-archive availability |

## Adding a new reproduction

1. Read the target paper's evaluation section carefully. Note:
   - Exact dataset + split + sample list.
   - View count / resolution / crop policy.
   - Scale alignment (metric? median? scale-and-shift?).
   - The metric name and the exact numerical value.
2. Write `reproductions/<short-name>.yaml`:
   - `model.name` + `kwargs` to match the paper's model variant + settings.
   - `dataset.name` + `kwargs` to match the paper's sample selection.
   - `tasks`, `scale_alignment`, `max_views` to match the protocol.
   - `paper_reference.primary_metric`, `.value`, `.tolerance_relative`.
   - `paper_reference.citation` — point a reader at the exact table/line.
3. For sample-level reproducibility, commit a `<short-name>.samples.txt`
   listing sample IDs in evaluation order and reference it from the YAML.
4. On the first successful run, pin the observed value in the YAML's
   `paper_reference.value` (if not already known from the paper) and the
   final `tolerance_relative`.

## Why tolerances

Bitwise reproducibility on CUDA is not possible for most current foundation
models — mixed-precision and cuDNN autotune introduce run-to-run noise. We
therefore express agreement as a **relative** tolerance on the primary
metric (default ±5%). If a run falls outside tolerance, investigate:

- Coordinate-system drift (the `conventions.py` assertions should catch most).
- Resolution / resize interpolation differences.
- Depth vs disparity vs inverse-depth confusion in the adapter.
- Scale alignment mode mismatch.

These failure modes are tracked as known traps in `plan.md § 9`.
