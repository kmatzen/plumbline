"""Reproductions: pinned paper-number configs.

A reproduction is a YAML file under ``reproductions/<name>.yaml`` declaring:

- Which model + version + preprocessing knobs to use.
- Which dataset + split + sample list.
- Which tasks, scale alignment, view count, resolution.
- The published reference: metric, value, and tolerance.

Running a reproduction executes :func:`~plumbline.runner.evaluate` with those
settings and compares the primary metric against the published value, within
tolerance.
"""

from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plumbline.cache import PredictionCache
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.report import Report
from plumbline.runner import evaluate

__all__ = [
    "REPRODUCTIONS_DIR",
    "ReproductionResult",
    "load_reproduction_config",
    "run_reproduction",
]

REPRODUCTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "reproductions"


@dataclass
class ReproductionResult:
    name: str
    report: Report
    primary_metric: str
    observed: float
    published: float
    tolerance_relative: float
    paper_match: bool | None
    notes: str = ""

    def to_markdown(self) -> str:
        md = self.report.to_markdown()
        md += "\n## Reproduction check\n\n"
        md += f"- Primary metric: `{self.primary_metric}`\n"
        md += f"- Observed: {self.observed:.4f}\n"
        md += f"- Published: {self.published:.4f}\n"
        md += f"- Tolerance (relative): {self.tolerance_relative:.2%}\n"
        if self.paper_match is not None:
            md += f"- Match: **{'yes' if self.paper_match else 'no'}**\n"
        if self.notes:
            md += f"\n> {self.notes}\n"
        return md


def load_reproduction_config(name: str) -> dict[str, Any]:
    """Load a reproduction YAML by short name (e.g. ``vggt-paper-scannet-depth``)."""
    path = _find_config(name)
    if path is None:
        raise FileNotFoundError(
            f"No reproduction config for '{name}'. Looked under {REPRODUCTIONS_DIR} "
            f"and the installed package."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_reproduction(name: str, *, output: Path | None = None) -> ReproductionResult:
    cfg = load_reproduction_config(name)

    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"model '{model_name}' not registered")
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"dataset '{dataset_name}' not registered")

    model_cls = MODEL_REGISTRY[model_name]
    dataset_cls = DATASET_REGISTRY[dataset_name]

    model_kwargs: dict[str, Any] = {"device": cfg.get("device", "cuda:0")}
    model_kwargs.update(cfg["model"].get("kwargs", {}))
    model = model_cls(**model_kwargs)

    dataset_kwargs: dict[str, Any] = dict(cfg["dataset"].get("kwargs", {}))
    if "split" in cfg["dataset"]:
        dataset_kwargs.setdefault("split", cfg["dataset"]["split"])
    dataset = dataset_cls(**dataset_kwargs)

    subset_n = cfg.get("subset")
    if subset_n:
        dataset = dataset.subset(int(subset_n))

    report = evaluate(
        model=model,
        dataset=dataset,
        tasks=list(cfg["tasks"]),
        scale_alignment=cfg.get("scale_alignment", "median"),
        max_views=int(cfg.get("max_views", 8)),
        device=cfg.get("device", "cuda:0"),
        cache=PredictionCache(cfg.get("cache_dir")) if cfg.get("cache_dir") else None,
    )

    paper = cfg.get("paper_reference", {})
    primary_metric = paper.get("primary_metric") or next(iter(report.aggregate_metrics))
    observed = float(report.aggregate_metrics.get(primary_metric, float("nan")))
    published = float(paper.get("value", float("nan")))
    tolerance = float(paper.get("tolerance_relative", 0.05))

    match: bool | None
    if published == published and observed == observed:  # both non-NaN
        match = abs(observed - published) / max(abs(published), 1e-8) <= tolerance
    else:
        match = None

    result = ReproductionResult(
        name=name,
        report=report,
        primary_metric=primary_metric,
        observed=observed,
        published=published,
        tolerance_relative=tolerance,
        paper_match=match,
        notes=cfg.get("notes", ""),
    )

    if output:
        report.save_json(output)
    return result


def _find_config(name: str) -> Path | None:
    candidates = [
        REPRODUCTIONS_DIR / f"{name}.yaml",
        REPRODUCTIONS_DIR / f"{name.replace('-', '_')}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    try:
        pkg = resources.files("plumbline").parent / "reproductions"  # type: ignore[attr-defined]
        for suffix in (f"{name}.yaml", f"{name.replace('-', '_')}.yaml"):
            candidate = pkg / suffix
            if candidate.is_file():  # type: ignore[attr-defined]
                return Path(str(candidate))
    except Exception:
        pass
    return None
