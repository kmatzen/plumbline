# Session handoff — pick up here

One-page pointer for the next GPU session. Per-session status lives in
the docs below; this doc only exists to route you there.

## Where to look

| For | Read |
|---|---|
| Paper-match status matrix (live) | `REPRODUCTIONS.md` |
| Open issues + next-session priorities | `docs/DISCREPANCIES.md` (single source — start with § Open issues at a glance) |
| GPU bring-up (human) | `GPU_RUNBOOK.md` |
| GPU bring-up (Claude-Code agent) | `docs/AGENT_GPU_RUNBOOK.md` (includes hard constraints + per-adapter extras) |
| Last session's full run log | `docs/runs/` (most recent dated file) |
| Architecture / extension guide | `docs/ARCHITECTURE.md` |
| Citation-audit findings | `reproductions/AUDIT.md` |

## S3 cache layout

`s3://plumbline-bench/` — 54 GB + 49.7 GB predictions:

- `datasets/` — source datasets (~12 GB)
- `hf-cache/` — HF model weights (~35 GB)
- `torch-hub-cache/` — Metric3D-v2 torch.hub (~7 GB)
- `predictions/<model>/<hash>/<dataset>/` — cached predictions for
  cheap re-scoring (~50 GB; pull selectively)
- `runs/<ts>/` — per-session results + logs + reports

Session token: run `scripts/gpu_box_session_token.sh` on laptop, paste
the 4 `export` lines on the rental box. 12 h validity.
