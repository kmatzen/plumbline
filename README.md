# plumbline

A reproducible evaluation harness for 3D geometric foundation models — think
`lm-evaluation-harness`, but for models like VGGT, Depth Anything 3, MASt3R,
Metric3Dv2, and Depth Pro.

**Status:** v0.1 in development. API will change.

## Goals

1. Run any supported model against any supported dataset with one command.
2. Reproduce the numbers from published papers, within a documented tolerance.
3. Impose canonical conventions (OpenCV camera frame, `world_from_camera`
   extrinsics, sRGB images) so models and datasets compose without leakage.

## Install

```bash
uv pip install -e ".[models]"
```

Or, for CPU-only development:

```bash
uv sync
```

## Quickstart

```bash
plumbline list-models
plumbline list-datasets
plumbline run --model depth-anything-v2 --dataset sintel --tasks mono_depth
```

## Reproduce a published number

```bash
plumbline reproduce vggt-paper-scannet-depth
```

## Documentation

- [`plan.md`](./plan.md) — full spec and roadmap.
- [`GPU_RUNBOOK.md`](./GPU_RUNBOOK.md) — how to run on a rented GPU.
- [`REPRODUCTIONS.md`](./REPRODUCTIONS.md) — paper-number configs and
  tolerances.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to add a model or dataset.

## License

Apache-2.0.
