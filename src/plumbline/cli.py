"""``plumbline`` CLI.

Uses typer (click under the hood). Kept thin: it's a view over
:mod:`~plumbline.runner` plus registry lookups.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from plumbline._version import __version__
from plumbline.cache import PredictionCache, default_cache_dir
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.report import Report
from plumbline.runner import evaluate

__all__ = ["app"]

app = typer.Typer(
    name="plumbline",
    help="Reproducible evaluation harness for 3D geometric foundation models.",
    no_args_is_help=False,
    add_completion=False,
)

console = Console()


def _eager_import_adapters() -> None:
    """Import built-in adapter modules so they register with the registry.

    Missing optional deps are swallowed — a missing torch or transformers
    should not block ``list-models`` or CLI help.
    """
    for mod in (
        "plumbline.models.depth_anything_v2",
        "plumbline.models.metric3d_v2",
        "plumbline.models.mast3r",
        "plumbline.models.vggt",
        "plumbline.models.depth_anything_3",
        "plumbline.datasets.sintel",
        "plumbline.datasets.scannet",
        "plumbline.datasets.eth3d",
    ):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # pragma: no cover — depends on optional deps
            console.print(f"[yellow]note:[/yellow] could not import {mod}: {exc}", soft_wrap=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"plumbline {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("list-models")
def list_models() -> None:
    """List registered model adapters."""
    _eager_import_adapters()
    table = Table(title="Registered models")
    table.add_column("name")
    table.add_column("tasks")
    table.add_column("is_metric")
    table.add_column("min/max views")
    for name in sorted(MODEL_REGISTRY):
        cls = MODEL_REGISTRY[name]
        caps = getattr(cls, "capabilities", None)
        if caps is None:
            table.add_row(name, "?", "?", "?")
            continue
        tasks = ", ".join(sorted(caps.tasks))
        maxv = "inf" if caps.max_views == float("inf") else str(int(caps.max_views))
        table.add_row(name, tasks, str(caps.is_metric), f"{caps.min_views} / {maxv}")
    console.print(table)


@app.command("list-datasets")
def list_datasets() -> None:
    """List registered datasets."""
    _eager_import_adapters()
    table = Table(title="Registered datasets")
    table.add_column("name")
    table.add_column("class")
    for name in sorted(DATASET_REGISTRY):
        cls = DATASET_REGISTRY[name]
        table.add_row(name, f"{cls.__module__}.{cls.__name__}")
    console.print(table)


@app.command("run")
def run_cmd(
    model: str = typer.Option(..., "--model", help="Registered model name."),
    dataset: str = typer.Option(..., "--dataset", help="Registered dataset name."),
    tasks: str = typer.Option("mono_depth", "--tasks", help="Comma-separated task list."),
    split: str | None = typer.Option(None, "--split", help="Dataset split."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Path to dataset root."),
    subset: int | None = typer.Option(None, "--subset", help="Evaluate on a subset of N samples."),
    scale_alignment: str = typer.Option(
        "median",
        "--scale-alignment",
        help="One of: none, median, lstsq, scale_shift.",
    ),
    max_views: int = typer.Option(8, "--max-views", help="Per-sample view cap."),
    device: str = typer.Option("cuda:0", "--device", help="Inference device."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON report to this path."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Override prediction cache dir."
    ),
) -> None:
    """Evaluate a model on a dataset."""
    _eager_import_adapters()

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    model_cls = _require(MODEL_REGISTRY, model, kind="model")
    dataset_cls = _require(DATASET_REGISTRY, dataset, kind="dataset")

    model_instance = model_cls(device=device)

    ds_kwargs: dict[str, object] = {}
    if split is not None:
        ds_kwargs["split"] = split
    if data_root is not None:
        ds_kwargs["root"] = data_root
    dataset_instance = dataset_cls(**ds_kwargs)

    if subset is not None:
        dataset_instance = dataset_instance.subset(subset)

    cache = PredictionCache(cache_dir) if cache_dir else PredictionCache()

    report = evaluate(
        model=model_instance,
        dataset=dataset_instance,
        tasks=task_list,
        scale_alignment=scale_alignment,
        max_views=max_views,
        device=device,
        cache=cache,
    )

    console.print(report.to_markdown())
    if output:
        report.save_json(output)
        console.print(f"[green]wrote[/green] {output}")


@app.command("reproduce")
def reproduce(
    name: str = typer.Argument(..., help="Reproduction name, e.g. vggt-paper-scannet-depth."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON report to this path."
    ),
) -> None:
    """Reproduce a published paper number from a pinned config."""
    from plumbline.reproduce import run_reproduction

    result = run_reproduction(name, output=output)
    console.print(result.to_markdown())
    if result.paper_match is not None:
        status = "[green]MATCH[/green]" if result.paper_match else "[red]MISMATCH[/red]"
        console.print(f"Paper-number check: {status}")


@app.command("report")
def report_cmd(
    path: Path = typer.Option(..., "--json", help="Path to a plumbline JSON report."),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown | json"),
) -> None:
    """Re-render a saved report."""
    report = Report.load_json(path)
    if format == "markdown":
        console.print(report.to_markdown())
    elif format == "json":
        typer.echo(report.to_json())
    else:
        raise typer.BadParameter(f"Unknown format: {format}")


@app.command("clear-cache")
def clear_cache(
    model: str | None = typer.Option(None, "--model", help="Only clear this model's cache."),
    dataset: str | None = typer.Option(None, "--dataset", help="Only clear this dataset's cache."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Cache root override."),
) -> None:
    """Remove cached predictions."""
    cache = PredictionCache(cache_dir) if cache_dir else PredictionCache()
    removed = cache.clear(model=model, dataset=dataset)
    console.print(f"[cyan]removed {removed} cached entries[/cyan] from {cache.predictions_dir}")


@app.command("cache-info")
def cache_info(
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Cache root override."),
) -> None:
    """Print cache location and size."""
    root = cache_dir if cache_dir else default_cache_dir()
    pred_dir = root / "predictions"
    if not pred_dir.exists():
        console.print(f"[yellow]no cache yet[/yellow] at {pred_dir}")
        return
    total = 0
    count = 0
    for f in pred_dir.rglob("*.npz"):
        total += f.stat().st_size
        count += 1
    console.print(f"{count} entries, {total / 1024 / 1024:.1f} MiB at {pred_dir}")


def _require(registry: dict[str, Any], key: str, *, kind: str) -> type:
    if key not in registry:
        options = ", ".join(sorted(registry))
        raise typer.BadParameter(f"Unknown {kind} '{key}'. Known: {options}")
    return registry[key]


# Expose for module-level scripts.
def main() -> None:  # pragma: no cover
    app()


_ = json  # kept for potential future use in this module
