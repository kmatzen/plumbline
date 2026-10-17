"""Run the authors' external EgoStatic supplement without vendoring its code.

This evaluates supplied sample poses, not new tracker inference or the complete
EgoStatic benchmark. The only compatibility shim restores NumPy's old np.int
alias. All sample outputs, logs, dependencies, and source hashes are retained.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MODELS = ("droid_slam", "colmap", "orbslam2", "orbslam3", "tartanvo", "particle_sfm", "monodepth2")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(source: Path, output: Path, models=MODELS):
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("preserve prior demo executions")
    if any(model not in MODELS for model in models):
        raise ValueError("unknown demo method")
    output.mkdir(parents=True)
    inputs = {
        str(p.relative_to(source)): digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".npy", ".txt", ".yml", ".md"}
    }
    record = {
        "status": "running",
        "source_url": "https://papers.neurips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Supplemental-Datasets_and_Benchmarks.zip",
        "supplement_sha256": "c3abb2a7acee336449ae5ea6f7c40fa17b386a801e1a6dc1fe9473f09736bf74",
        "nested_archive_sha256": "f188c8ac4f09f8aa47bb3b786b60a4ed0d0d791117fc3c01434e6f4fc570f8eb",
        "script_sha256": digest(Path(__file__)),
        "source_files_sha256": inputs,
        "compatibility": "np.int temporarily aliases built-in int; external source files unchanged",
        "scope": "official one-sequence sample-pose demo; not full benchmark reproduction or new inference",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "methods": [],
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    original_directory, original_argv, original_path = Path.cwd(), sys.argv[:], sys.path[:]
    had_alias = "int" in np.__dict__
    old_alias = np.__dict__.get("int")
    np.int = int
    try:
        os.chdir(source)
        sys.path.insert(0, str(source))
        for model in models:
            entry = {"model": model, "status": "running"}
            record["methods"].append(entry)
            save()
            captured = io.StringIO()
            start = time.perf_counter()
            try:
                sys.argv = [str(source / "ego4d_evaluation.py"), "--model", model]
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    namespace = runpy.run_path(
                        str(source / "ego4d_evaluation.py"), run_name="__main__"
                    )
                metric = serializable(namespace.get("log_metric", {}))
                if not metric or "Average ORE is" not in captured.getvalue():
                    raise ValueError("demo did not produce a completed metric")
                entry.update({"status": "complete", "log_metric": metric})
            except Exception as error:
                entry.update(
                    {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
                )
            entry["elapsed_seconds"] = time.perf_counter() - start
            log = output / f"{model}.log"
            log.write_text(captured.getvalue())
            entry["log_sha256"] = digest(log)
            save()
            print(model, entry["status"], flush=True)
        import scipy

        record["scipy_version"] = scipy.__version__
        record["status"] = (
            "complete"
            if all(m["status"] == "complete" for m in record["methods"])
            else "completed_with_failures"
        )
    finally:
        os.chdir(original_directory)
        sys.argv, sys.path = original_argv, original_path
        if had_alias:
            np.int = old_alias
        else:
            delattr(np, "int")
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    run(args.source, args.output, args.models)
