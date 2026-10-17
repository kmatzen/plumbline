"""Pinned CoTracker3 development probe on frozen image-only query locations.

Run on the research GPU. No checkpoint download, camera-dependent point
selection, or automatic retry. Prefix observations are computed separately so
the early point fit need not use future video frames through an offline tracker.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_queries(baseline):
    slots, queries = [], []
    for slot, track in enumerate(baseline["tracks"]):
        xy = track[0]["xy"]
        if xy is not None:
            slots.append(slot)
            queries.append([0, *xy])
    if not queries or not np.isfinite(queries).all():
        raise ValueError("finite nonempty frozen queries required")
    return slots, np.asarray(queries, dtype=np.float32)


def run(source, root, baseline_path, repo, checkpoint, output):
    if output.exists():
        raise ValueError("preserve existing attempts")
    baseline = json.loads(baseline_path.read_text())
    manifest = json.loads(source.read_text())
    if manifest.get("status") != "complete" or baseline["methods"][0]["manifest_sha256"] != digest(
        source
    ):
        raise ValueError("source and frozen query provenance mismatch")
    slots, queries = frozen_queries(baseline)
    output.mkdir(parents=True)
    record = {
        "status": "running",
        "source_manifest_sha256": digest(source),
        "baseline_sha256": digest(baseline_path),
        "script_sha256": digest(Path(__file__)),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "repository_status": subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip(),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint": str(checkpoint),
        "source_files_sha256": {
            str(p.relative_to(repo)): digest(p) for p in sorted((repo / "cotracker").rglob("*.py"))
        },
        "slots": slots,
        "queries_txy": queries.tolist(),
        "phases": [],
        "scope": "candidate-independent image tracks; not ground truth or static-scene certification",
        "fit_policy": "separate first-11-frame inference; do not fit from full-video early tracks",
        "predictor_policy": "upstream offline predictor defaults: support grid, visibility threshold, query correction retained",
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    try:
        import torch
        from PIL import Image

        sys.path.insert(0, str(repo))
        from cotracker.predictor import CoTrackerPredictor

        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        record["torch_version"] = torch.__version__
        record["gpu"] = torch.cuda.get_device_name()
        frames = []
        for frame in manifest["frames"]:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != frame["file_sha256"]:
                raise ValueError("source image hash/path mismatch")
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))
        frames = np.stack(frames)
        record["input_rgb_sha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
        record["shape_thwc"] = list(frames.shape)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = (
            CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60)
            .eval()
            .cuda()
        )
        query = torch.from_numpy(queries)[None].cuda()
        for name, count in (("prefix", 11), ("full", len(frames))):
            phase = {"name": name, "frames": count, "status": "running"}
            record["phases"].append(phase)
            save()
            video = torch.from_numpy(frames[:count].copy()).permute(0, 3, 1, 2)[None].float().cuda()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                tracks, visibility = model(video, queries=query, backward_tracking=False)
            torch.cuda.synchronize()
            phase.update(
                {
                    "seconds": time.perf_counter() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            xy, visible = tracks[0].cpu().numpy(), visibility[0].cpu().numpy()
            if (
                xy.shape != (count, len(slots), 2)
                or visible.shape != (count, len(slots))
                or not np.isfinite(xy).all()
            ):
                raise ValueError("unexpected track output")
            np.savez_compressed(
                output / f"{name}.npz",
                tracks=xy,
                visibility=visible,
                slots=np.asarray(slots),
                queries=queries,
            )
            phase.update(
                {"status": "complete", "prediction_sha256": digest(output / f"{name}.npz")}
            )
            save()
            del video, tracks, visibility
        record["status"] = "complete"
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        save()
        raise
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "root", "baseline", "repo", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.source, args.root, args.baseline, args.repo, args.checkpoint, args.output)
