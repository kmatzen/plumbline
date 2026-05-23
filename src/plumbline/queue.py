"""GPU run queue — a declarative, ordered backlog of reproductions awaiting a GPU.

The queue lives in ``reproductions/gpu_queue.yaml``. Each job points at an
existing reproduction (by its ``name:``) and carries the operator context
needed to provision a rental box: which datasets to stage, disk footprint,
expected wall-time, required env vars, and the paper target the run is checked
against.

This module is the executable counterpart to ``GPU_RUNBOOK.md``:

- :func:`load_queue` parses + validates the file (every ``name`` must resolve
  to a real reproduction config).
- :func:`run_queue` executes the ``pending`` jobs in priority order via
  :func:`plumbline.reproduce.run_reproduction` and records, per job, whether it
  was a paper MATCH / MISMATCH / informational run / ERROR.

It never mutates reproduction YAMLs and never invents numbers — a failed run is
a recorded finding, per the runbook's hard constraints. Job ``status`` in the
queue file is hand-maintained; ``run_queue`` reads it but does not write it back.
"""

from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from plumbline.reproduce import REPRODUCTIONS_DIR, _find_config

__all__ = [
    "QUEUE_PATH",
    "QueueJob",
    "QueueRunRecord",
    "load_queue",
    "run_queue",
]

QUEUE_PATH = REPRODUCTIONS_DIR / "gpu_queue.yaml"

_VALID_STATUS = {"pending", "blocked", "done"}


@dataclass
class QueueJob:
    """One entry in the GPU queue.

    ``name`` must resolve to an existing reproduction config. The remaining
    fields are operator metadata (used for provisioning + the listing view);
    they are descriptive and do not affect how the reproduction runs.
    """

    name: str
    status: str = "pending"
    priority: int = 100
    paper_target: str = ""
    primary_metric: str = ""
    datasets: list[str] = field(default_factory=list)
    data_footprint_gb: float = 0.0
    est_wall_min: int = 0
    requires_env: list[str] = field(default_factory=list)
    extras: str = ""
    blocked_on: str = ""
    notes: str = ""

    @property
    def runnable(self) -> bool:
        return self.status == "pending"


@dataclass
class QueueRunRecord:
    """Outcome of attempting one queue job."""

    name: str
    # one of: "match", "mismatch", "info", "error", "skipped"
    outcome: str
    observed: float | None = None
    published: float | None = None
    primary_metric: str = ""
    error: str = ""


def _queue_file() -> Path:
    """Resolve the queue file path, falling back to the installed package."""
    if QUEUE_PATH.exists():
        return QUEUE_PATH
    try:
        pkg = resources.files("plumbline").parent / "reproductions" / "gpu_queue.yaml"  # type: ignore[attr-defined]
        if pkg.is_file():
            return Path(str(pkg))
    except Exception:
        pass
    return QUEUE_PATH  # let the caller raise a clear FileNotFoundError


def load_queue(path: Path | None = None) -> list[QueueJob]:
    """Load + validate the GPU queue, sorted by (priority, file order).

    Raises ``ValueError`` if a job has an unknown status or a ``name`` that
    doesn't resolve to a reproduction config — a broken queue should fail
    loudly, not silently skip work.
    """
    p = path or _queue_file()
    if not p.exists():
        raise FileNotFoundError(f"GPU queue not found at {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_jobs = data.get("jobs") or []

    jobs: list[QueueJob] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_jobs):
        if "name" not in raw:
            raise ValueError(f"queue job #{i} has no 'name'")
        name = str(raw["name"])
        if name in seen:
            raise ValueError(f"duplicate queue job name: {name!r}")
        seen.add(name)

        status = str(raw.get("status", "pending"))
        if status not in _VALID_STATUS:
            raise ValueError(
                f"queue job {name!r}: invalid status {status!r} "
                f"(expected one of {sorted(_VALID_STATUS)})"
            )
        if _find_config(name) is None:
            raise ValueError(
                f"queue job {name!r}: no reproduction config found. "
                f"Looked under {REPRODUCTIONS_DIR}."
            )

        known = set(QueueJob.__dataclass_fields__)
        kwargs: dict[str, Any] = {k: v for k, v in raw.items() if k in known}
        kwargs["name"] = name
        kwargs["status"] = status
        jobs.append(QueueJob(**kwargs))

    jobs.sort(key=lambda j: (j.priority,))
    return jobs


def run_queue(
    *,
    only: str | None = None,
    output: Path | None = None,
) -> list[QueueRunRecord]:
    """Execute the ``pending`` queue jobs in priority order.

    ``only`` restricts to a single job by name (it is run even though the
    listing would otherwise require ``pending`` — but its declared status must
    still be ``pending``; ``blocked``/``done`` jobs are never auto-run).

    Each job is run via :func:`plumbline.reproduce.run_reproduction`. Exceptions
    (missing data, OOM, adapter import failure) are caught and recorded as
    ``error`` so one bad job doesn't abort the rest of the queue.
    """
    import json

    from plumbline.reproduce import run_reproduction

    jobs = [j for j in load_queue() if j.runnable]
    if only is not None:
        jobs = [j for j in jobs if j.name == only]

    records: list[QueueRunRecord] = []
    for job in jobs:
        try:
            result = run_reproduction(job.name)
        except Exception as exc:  # queue must survive any single job's failure
            records.append(
                QueueRunRecord(
                    name=job.name,
                    outcome="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if result.paper_match is True:
            outcome = "match"
        elif result.paper_match is False:
            outcome = "mismatch"
        else:
            outcome = "info"
        records.append(
            QueueRunRecord(
                name=job.name,
                outcome=outcome,
                observed=result.observed,
                published=result.published,
                primary_metric=result.primary_metric,
            )
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0.0",
            "records": [
                {
                    "name": r.name,
                    "outcome": r.outcome,
                    "observed": r.observed,
                    "published": r.published,
                    "primary_metric": r.primary_metric,
                    "error": r.error,
                }
                for r in records
            ],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return records
