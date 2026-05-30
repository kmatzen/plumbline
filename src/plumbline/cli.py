"""``plumbline`` CLI.

Uses typer (click under the hood). Kept thin: it's a view over
:mod:`~plumbline.runner` plus registry lookups.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from plumbline._discover import register_builtin_adapters
from plumbline._version import __version__
from plumbline.cache import PredictionCache, default_cache_dir
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.install import INSTALL_SPECS, check, install_hint, install_plan, spec_for
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


def _parse_kv_value(v: str) -> object:
    """Parse a ``KEY=VALUE`` right-hand side into an int/float/bool/str.

    Bare tokens are tried in order: int, float, ``true``/``false``, plain
    string. Quoted strings lose their outer quotes. ``none``/``null`` →
    Python ``None``.
    """
    s = v.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    lo = s.lower()
    if lo in ("none", "null"):
        return None
    if lo == "true":
        return True
    if lo == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _eager_import_adapters() -> None:
    """Import built-in adapter modules so they register with the registry.

    Missing optional deps are swallowed — a missing torch or transformers
    should not block ``list-models`` or CLI help. Failures surface as
    yellow notes in the console.
    """
    failures = register_builtin_adapters()
    for mod, exc in failures:
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
    dataset_kwargs: list[str] = typer.Option(
        [],
        "--dataset-kwargs",
        help=(
            "Extra kwargs for the dataset constructor as KEY=VALUE pairs "
            "(e.g. --dataset-kwargs views_per_sample=8 --dataset-kwargs "
            "depth_field=raw). Values are parsed as int / float / bool / str."
        ),
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
    for kv in dataset_kwargs:
        if "=" not in kv:
            raise typer.BadParameter(f"--dataset-kwargs expects KEY=VALUE; got {kv!r}")
        k, _, v = kv.partition("=")
        ds_kwargs[k.strip()] = _parse_kv_value(v)
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


@app.command("queue")
def queue_cmd(
    run: bool = typer.Option(
        False,
        "--run",
        help="Execute the pending jobs (needs a GPU + staged data). Default: list only.",
    ),
    name: str | None = typer.Option(
        None, "--name", help="Restrict to a single job by reproduction name."
    ),
    include_blocked: bool = typer.Option(
        False, "--include-blocked", help="Also show blocked jobs in the listing."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write a JSON run summary (with --run)."
    ),
) -> None:
    """Show or run the GPU job queue (``reproductions/gpu_queue.yaml``).

    Without ``--run`` this is a planning view: pending/blocked jobs, their
    paper targets, data footprints, and required env vars. With ``--run`` it
    executes the pending jobs in priority order and reports MATCH / MISMATCH /
    INFO / ERROR for each — the executable counterpart to ``GPU_RUNBOOK.md``.
    """
    from plumbline.queue import load_queue, run_queue

    if run:
        _eager_import_adapters()
        records = run_queue(only=name, output=output)
        if not records:
            console.print("[yellow]no pending jobs to run[/yellow]")
            raise typer.Exit()
        table = Table(title="GPU queue — run results")
        table.add_column("job")
        table.add_column("outcome")
        table.add_column("metric")
        table.add_column("observed")
        table.add_column("published")
        style = {
            "match": "green",
            "mismatch": "red",
            "info": "cyan",
            "error": "yellow",
        }
        n_match = 0
        for r in records:
            if r.outcome == "match":
                n_match += 1
            obs = f"{r.observed:.4f}" if r.observed is not None else "—"
            pub = (
                f"{r.published:.4f}"
                if r.published is not None and r.published == r.published
                else "—"
            )
            detail = r.error if r.outcome == "error" else r.outcome
            table.add_row(
                r.name,
                f"[{style.get(r.outcome, 'white')}]{detail}[/]",
                r.primary_metric or "—",
                obs,
                pub,
            )
        console.print(table)
        console.print(f"{n_match}/{len(records)} job(s) matched paper within tolerance.")
        if output:
            console.print(f"[green]wrote[/green] {output}")
        return

    jobs = load_queue()
    shown = jobs if include_blocked else [j for j in jobs if j.status != "blocked"]
    table = Table(title="GPU queue (reproductions/gpu_queue.yaml)")
    table.add_column("pri", justify="right")
    table.add_column("job")
    table.add_column("status")
    table.add_column("GB", justify="right")
    table.add_column("~min", justify="right")
    table.add_column("env")
    table.add_column("paper target")
    sstyle = {"pending": "green", "blocked": "yellow", "done": "dim"}
    for j in shown:
        table.add_row(
            str(j.priority),
            j.name,
            f"[{sstyle.get(j.status, 'white')}]{j.status}[/]",
            f"{j.data_footprint_gb:g}",
            str(j.est_wall_min),
            ",".join(j.requires_env),
            j.paper_target or "—",
        )
    console.print(table)
    n_pending = sum(1 for j in jobs if j.status == "pending")
    n_blocked = sum(1 for j in jobs if j.status == "blocked")
    n_done = sum(1 for j in jobs if j.status == "done")
    hidden = "" if include_blocked else f" ({n_blocked} blocked hidden; --include-blocked to show)"
    console.print(
        f"{n_pending} pending, {n_blocked} blocked, {n_done} done.{hidden} "
        f"Run with: [bold]plumbline queue --run[/bold]"
    )


def _run_steps(name: str, steps: list[str]) -> None:
    """Run the pip/git steps of an install plan via subprocess.

    ``uv pip install ...`` lines are rewritten to target the current
    interpreter explicitly (``uv pip install --python <sys.executable> ...``) so
    the deps land in the venv that is running plumbline, not whatever virtualenv
    ``uv`` would otherwise default to. ``uv`` is a standalone binary (installed
    via the astral.sh installer, per GPU_RUNBOOK.md), so it is invoked directly
    rather than as ``python -m uv`` — the latter only works if ``uv`` happens to
    be pip-installed into the venv, which it generally is not. ``git clone``
    lines run as-is. ``export ...`` lines are never executed (a child process
    can't mutate the parent's environment); they are printed instead.

    Env vars in tokens (e.g. ``$HOME`` in a clone's default destination) are
    expanded with ``os.path.expandvars`` — subprocess does not go through a
    shell, so without this a ``git clone ... $HOME/deps/foo`` step would create
    a literal ``$HOME`` directory under the cwd.
    """
    for step in steps:
        if step.startswith("export "):
            console.print(f"[yellow]set this yourself:[/yellow] {step}")
            continue
        # shlex.split so shell quoting in the plan (e.g. a quoted git URL) is
        # honored rather than surviving as literal quote chars in argv; then
        # expandvars so $HOME-style refs resolve as a shell would.
        tokens = [os.path.expandvars(t) for t in shlex.split(step)]
        if step.startswith("uv pip install"):
            # tokens == ["uv", "pip", "install", <args...>]; target the running venv.
            argv = ["uv", "pip", "install", "--python", sys.executable, *tokens[3:]]
        else:
            argv = tokens
        console.print(f"[cyan]$[/cyan] {' '.join(argv)}")
        # argv is built from the trusted INSTALL_SPECS registry, not user input.
        result = subprocess.run(argv, check=False)
        if result.returncode != 0:
            console.print(
                f"[red]install step failed[/red] for {name!r} (exit {result.returncode}): {step}"
            )
            raise typer.Exit(result.returncode)


@app.command("install")
def install_cmd(
    name: str | None = typer.Argument(None, help="Adapter to install (omit to list all)."),
    all_: bool = typer.Option(
        False, "--all", help="Install every pip/git adapter (skips clone adapters)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Actually run the pip/git steps (default: print the plan)."
    ),
    list_: bool = typer.Option(False, "--list", help="List all adapters and how to install them."),
) -> None:
    """Show or run adapter install plans (source: ``plumbline.install``).

    With no NAME (or ``--list``) this prints a table of every adapter, its kind,
    and a one-line how-to. With NAME it prints that adapter's ordered install
    plan; add ``--yes`` to actually run the pip/git steps into the current venv.
    For ``clone`` adapters ``--yes`` runs the ``git clone`` + extra-deps steps
    and then prints the ``export`` lines you must add to your shell (env is never
    persisted). ``--all`` iterates the pip + git adapters.
    """
    if all_:
        targets = [n for n, s in INSTALL_SPECS.items() if s.kind in ("pypi", "git")]
        for n in sorted(targets):
            console.print(f"[bold]{n}[/bold] ({spec_for(n).kind})")
            steps = install_plan(n)
            if yes:
                _run_steps(n, steps)
            else:
                for step in steps:
                    console.print(f"  {step}")
        if not yes:
            console.print("\nRe-run with [bold]--yes[/bold] to execute these steps.")
        return

    if name is None or list_:
        table = Table(title="Adapter install plans (source: src/plumbline/install.py)")
        table.add_column("adapter")
        table.add_column("kind")
        table.add_column("how to install")
        for n in sorted(INSTALL_SPECS):
            spec = INSTALL_SPECS[n]
            table.add_row(n, spec.kind, spec.how_to())
        console.print(table)
        console.print(
            "Run [bold]plumbline install <name>[/bold] for the full plan, "
            "[bold]--yes[/bold] to execute, or [bold]plumbline doctor[/bold] to check status."
        )
        return

    if name not in INSTALL_SPECS:
        options = ", ".join(sorted(INSTALL_SPECS))
        raise typer.BadParameter(f"Unknown adapter '{name}'. Known: {options}")

    spec = spec_for(name)
    steps = install_plan(name)
    if not steps:
        console.print(f"[green]{name}[/green] {spec.how_to()} — nothing to install.")
        return

    if not yes:
        console.print(f"[bold]install plan for {name}[/bold] ({spec.kind}):")
        for step in steps:
            console.print(f"  {step}")
        console.print("\nRe-run with [bold]--yes[/bold] to execute these steps.")
        return

    _run_steps(name, steps)
    if spec.kind == "clone":
        console.print(
            f"\n[green]cloned[/green] {name}. Add these to your shell so the adapter "
            "can find the clone:"
        )
        for step in steps:
            if step.startswith("export "):
                console.print(f"  {step}")
    else:
        console.print(f"[green]installed[/green] {name}.")


@app.command("doctor")
def doctor_cmd(
    name: str | None = typer.Argument(None, help="Adapter to check (omit to check all)."),
) -> None:
    """Check which adapters look installed (CI-gateable).

    Runs the registry's ``check`` for each requested adapter and prints an
    OK / MISSING table with the fix hint. Exits nonzero if any *requested*
    adapter is MISSING, so CI can gate on a specific adapter being present.
    """
    if name is not None and name not in INSTALL_SPECS:
        options = ", ".join(sorted(INSTALL_SPECS))
        raise typer.BadParameter(f"Unknown adapter '{name}'. Known: {options}")

    names = [name] if name is not None else sorted(INSTALL_SPECS)
    table = Table(title="Adapter doctor")
    table.add_column("adapter")
    table.add_column("status")
    table.add_column("fix")
    n_missing = 0
    for n in names:
        ok, _detail = check(n)
        if ok:
            status = "[green]OK[/green]"
            fix = "—"
        else:
            status = "[red]MISSING[/red]"
            fix = install_hint(n)
            n_missing += 1
        table.add_row(n, status, fix)
    console.print(table)
    console.print(f"{len(names) - n_missing}/{len(names)} OK, {n_missing} missing.")
    if n_missing:
        raise typer.Exit(1)


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


@app.command("make-samples")
def make_samples(
    dataset: str = typer.Option(..., "--dataset", help="Registered dataset name."),
    output: Path = typer.Option(..., "--output", "-o", help="Write sample IDs here."),
    split: str | None = typer.Option(None, "--split", help="Dataset split."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Path to dataset root."),
    subset: int | None = typer.Option(
        None, "--subset", help="Take the first N after the dataset's deterministic ordering."
    ),
) -> None:
    """Write a reproduction sample-list file (``sample_ids_file:`` target).

    Materializes the exact sample IDs a dataset would yield under the given
    kwargs, so a reproduction YAML can pin them and stay stable across
    manifest re-scans.
    """
    import datetime as _dt

    _eager_import_adapters()

    dataset_cls = _require(DATASET_REGISTRY, dataset, kind="dataset")
    ds_kwargs: dict[str, Any] = {}
    if split is not None:
        ds_kwargs["split"] = split
    if data_root is not None:
        ds_kwargs["root"] = data_root
    dataset_instance = dataset_cls(**ds_kwargs)

    if subset is not None:
        dataset_instance = dataset_instance.subset(subset)

    sample_ids = [sample.sample_id for sample in dataset_instance]

    output.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    header = [
        f"# Generated by plumbline make-samples on {now}",
        f"# plumbline: {__version__}",
        f"# dataset: {dataset}",
    ]
    if split is not None:
        header.append(f"# split: {split}")
    if data_root is not None:
        header.append(f"# data-root: {data_root}")
    if subset is not None:
        header.append(f"# subset: {subset}")
    header.append(f"# n_samples: {len(sample_ids)}")
    output.write_text("\n".join(header + sample_ids) + "\n", encoding="utf-8")
    console.print(
        f"[green]wrote[/green] {len(sample_ids)} sample ID(s) to {output}",
    )


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
