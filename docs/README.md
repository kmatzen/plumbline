# Documentation map

Minimal set of markdown docs. **Start here** for navigation; avoid duplicating
content that already lives in a canonical file.

## Core (read first)

| Doc | Purpose |
|-----|---------|
| [`../README.md`](../README.md) | Project overview, install |
| [`../REPRODUCTIONS.md`](../REPRODUCTIONS.md) | Paper-match matrix (live status) |
| [`../GPU_RUNBOOK.md`](../GPU_RUNBOOK.md) | GPU bring-up, queue, single-record diff, S3 |
| [`DISCREPANCIES.md`](DISCREPANCIES.md) | Outstanding-work tracker (open/parked/investigated D-numbers only) |
| [`CONFIDENCE_AUDIT.md`](CONFIDENCE_AUDIT.md) | Resolved understanding: where each off-paper gap lives (adapter / parsing / paper recipe / checkpoint), confirmed vs unknown, + per-paper trust |
| [`BLOCKED.md`](BLOCKED.md) | Index of fundamentally blocked cells |
| [`../reproductions/gpu_queue.yaml`](../reproductions/gpu_queue.yaml) | Machine-readable run queue |

## Extend the harness

| Doc | Purpose |
|-----|---------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Vocabulary, library API, adding adapters — **known traps** §10 |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Dev setup, PR expectations |
| [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) | Adapter vs upstream source fidelity |
| [`../reproductions/AUDIT.md`](../reproductions/AUDIT.md) | Paper citation / PDF audit per YAML |

## Blocked cells (`blocked/`)

One page per closed blocker. Linked from [`BLOCKED.md`](BLOCKED.md).

## Return handoffs (parked work, not duplicate of D-entries)

| Doc | When to open |
|-----|----------------|
| [`D29_DIODE_TABLE2_HANDOFF.md`](D29_DIODE_TABLE2_HANDOFF.md) | Native DIODE Table 2 outdoor gap |
| [`ETH3D_DAV2_TABLE2_HANDOFF.md`](ETH3D_DAV2_TABLE2_HANDOFF.md) | Native ETH3D Table 2 (D31/D33) |
| [`SINTEL_DAV2_TABLE2_HANDOFF.md`](SINTEL_DAV2_TABLE2_HANDOFF.md) | Native Sintel Table 2 (D32) |
| [`ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md`](ETH3D_DEPTH_PRO_TABLE1_HANDOFF.md) | Depth Pro ETH3D δ₁ (pending data) |
| [`DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md) | MoGe-lineage Table 2 archaeology (shared) |
| [`UPSTREAM_OUTREACH.md`](UPSTREAM_OUTREACH.md) | Issue-comment drafts |

## Removed (audit 2026-05-31)

These were redundant or session-only; content lives elsewhere:

- `plan.md` → traps in `ARCHITECTURE.md` §10; workflow in `GPU_RUNBOOK.md`
- `docs/GPU_BACKLOG_PLAN.md` → active work in `GPU_RUNBOOK.md` § Active work
- `docs/DEPTH_PRO_TABLE1_METRIC_EVAL.md` → `BLOCKED.md` + `blocked/DEPTH_PRO_*`
- `SESSION_2026-05-25.md`, `docs/runs/archive/20260421.md` → `DISCREPANCIES.md` + S3 JSONs
