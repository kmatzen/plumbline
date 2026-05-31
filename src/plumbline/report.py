"""Report: markdown + JSON output from an evaluation run.

The JSON schema is public API and is versioned. Future changes to the schema
bump ``SCHEMA_VERSION`` and — when breaking — add a migration note in
``REPRODUCTIONS.md``.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA_VERSION", "Report", "RunEnvironment", "SampleResult"]

SCHEMA_VERSION = "1.0.0"


@dataclass
class RunEnvironment:
    """Environment info captured for reproducibility."""

    python: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    plumbline_version: str = ""
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "plumbline_version": self.plumbline_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class SampleResult:
    """Per-sample metric values."""

    sample_id: str
    metrics: dict[str, float]
    skipped: bool = False
    skip_reason: str | None = None
    runtime_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metrics": self.metrics,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "runtime_ms": self.runtime_ms,
        }


@dataclass
class Report:
    """Structured result of an evaluation run."""

    model: str
    model_version: str
    dataset: str
    split: str
    tasks: list[str]
    scale_alignment: str
    aggregate_metrics: dict[str, float]
    per_sample: list[SampleResult] = field(default_factory=list)
    per_scene_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    n_total: int = 0
    n_evaluated: int = 0
    n_skipped: int = 0
    environment: RunEnvironment = field(default_factory=RunEnvironment)
    config_hash: str = ""
    notes: str = ""

    # -- JSON -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "model_version": self.model_version,
            "dataset": self.dataset,
            "split": self.split,
            "tasks": list(self.tasks),
            "scale_alignment": self.scale_alignment,
            "aggregate_metrics": dict(self.aggregate_metrics),
            "per_scene_metrics": {s: dict(m) for s, m in self.per_scene_metrics.items()},
            "per_sample": [s.to_dict() for s in self.per_sample],
            "n_total": self.n_total,
            "n_evaluated": self.n_evaluated,
            "n_skipped": self.n_skipped,
            "config_hash": self.config_hash,
            "environment": self.environment.to_dict(),
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=_default)

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> Report:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported report schema version {version!r}; expected {SCHEMA_VERSION!r}. "
                "Future releases may add migrations."
            )
        env = data.get("environment", {})
        return cls(
            model=data["model"],
            model_version=data.get("model_version", ""),
            dataset=data["dataset"],
            split=data.get("split", ""),
            tasks=list(data.get("tasks", [])),
            scale_alignment=data.get("scale_alignment", "none"),
            aggregate_metrics=dict(data.get("aggregate_metrics", {})),
            per_scene_metrics={s: dict(m) for s, m in data.get("per_scene_metrics", {}).items()},
            per_sample=[
                SampleResult(
                    sample_id=s["sample_id"],
                    metrics=dict(s.get("metrics", {})),
                    skipped=bool(s.get("skipped", False)),
                    skip_reason=s.get("skip_reason"),
                    runtime_ms=s.get("runtime_ms"),
                )
                for s in data.get("per_sample", [])
            ],
            n_total=int(data.get("n_total", 0)),
            n_evaluated=int(data.get("n_evaluated", 0)),
            n_skipped=int(data.get("n_skipped", 0)),
            environment=RunEnvironment(
                python=env.get("python", ""),
                platform=env.get("platform", ""),
                plumbline_version=env.get("plumbline_version", ""),
                torch_version=env.get("torch_version"),
                cuda_version=env.get("cuda_version"),
                gpu_name=env.get("gpu_name"),
                timestamp_utc=env.get("timestamp_utc", ""),
            ),
            config_hash=data.get("config_hash", ""),
            notes=data.get("notes", ""),
        )

    # -- Markdown ---------------------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# {self.model} on {self.dataset}")
        lines.append("")
        lines.append(f"**Split:** `{self.split}`  ")
        lines.append(f"**Tasks:** `{', '.join(self.tasks)}`  ")
        lines.append(f"**Scale alignment:** `{self.scale_alignment}`  ")
        lines.append(f"**Samples evaluated:** {self.n_evaluated} / {self.n_total}")
        if self.n_skipped:
            lines.append(f"**Skipped:** {self.n_skipped}")
        lines.append("")

        if self.aggregate_metrics:
            lines.append("## Aggregate metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for name in sorted(self.aggregate_metrics):
                val = self.aggregate_metrics[name]
                lines.append(f"| {name} | {_fmt(val)} |")
            lines.append("")

        if self.per_scene_metrics:
            lines.append("## Per-scene metrics")
            lines.append("")
            metric_names = sorted({k for m in self.per_scene_metrics.values() for k in m})
            lines.append("| Scene | " + " | ".join(metric_names) + " |")
            lines.append("| " + " | ".join(["---"] * (len(metric_names) + 1)) + " |")
            for scene in sorted(self.per_scene_metrics):
                row = self.per_scene_metrics[scene]
                cells = [_fmt(row.get(k, float("nan"))) for k in metric_names]
                lines.append(f"| {scene} | " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("## Environment")
        lines.append("")
        lines.append(f"- `plumbline` {self.environment.plumbline_version}")
        lines.append(f"- Python {self.environment.python} on {self.environment.platform}")
        if self.environment.torch_version:
            lines.append(f"- torch {self.environment.torch_version}")
        if self.environment.gpu_name:
            lines.append(
                f"- GPU: {self.environment.gpu_name} (CUDA {self.environment.cuda_version})"
            )
        lines.append(f"- Config hash: `{self.config_hash}`")
        lines.append(f"- Run at: {self.environment.timestamp_utc}")
        if self.notes:
            lines.append("")
            lines.append(f"> {self.notes}")
        return "\n".join(lines) + "\n"


def _default(obj: Any) -> Any:
    # Keep to_json() tolerant to numpy scalars showing up in metadata.
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def _fmt(value: float) -> str:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        # Consistent metric formatting: 4 decimal places, with small values
        # bumped to scientific.
        if 0 < abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)
