"""Fresh, pose-only DAGE run with full input hashes; never reuse prediction cache.

Run in the existing Anima DAGE environment with the original package on
PYTHONPATH. Output directory must be new. This is a development experiment,
not a held-out evaluation or a validated functional-DNF label.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(root: Path, checkpoint: Path, sequence: str, output: Path) -> None:
    if output.exists():
        raise ValueError("output exists; preserve previous attempts and choose a new directory")
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be a pinned local snapshot directory")
    if not (checkpoint / "model.pt").is_file():
        raise ValueError("checkpoint snapshot must contain model.pt")
    import torch

    from plumbline.datasets.tum_dynamics import TUMDynamicsDataset
    from plumbline.models.dage import DAGEAdapter

    output.mkdir(parents=True)
    manifest = {"status": "preparing", "sequence": sequence, "development_run": True}

    def save() -> None:
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    try:
        dataset = TUMDynamicsDataset(root=root, scenes=[sequence])
        # Rescan raw timestamp files rather than trusting a stale loader manifest.
        records = [r for r in dataset._scan() if r["scene"] == sequence]
        if len(records) != 1:
            raise ValueError("expected exactly one staged sequence")
        record = records[0]
        sample = dataset._load_sample(record)
        frames = []
        for index, (relative, gt_row) in enumerate(
            zip(record["image_paths"], record["gt_rows"], strict=True)
        ):
            path = root / relative
            frames.append(
                {
                    "path": relative,
                    "timestamp": path.stem,
                    "reference_timestamp": gt_row[0],
                    "file_sha256": digest_file(path),
                    "decoded_rgb_sha256": hashlib.sha256(
                        sample.images[index].tobytes()
                    ).hexdigest(),
                }
            )
        package_root = Path(inspect.getfile(DAGEAdapter)).parents[1]
        source_hashes = {
            str(p.relative_to(package_root)): digest_file(p)
            for p in sorted(package_root.rglob("*.py"))
        }
        manifest.update(
            {
                "status": "running",
                "frames": frames,
                "shape": list(sample.images.shape),
                "input_sha256": hashlib.sha256(sample.images.tobytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(sample.extrinsics_gt.tobytes()).hexdigest(),
                "timestamp_files": {
                    name: digest_file(root / sequence / name)
                    for name in ("rgb.txt", "groundtruth.txt")
                },
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_files": {
                    str(p.relative_to(checkpoint)): digest_file(p)
                    for p in sorted(checkpoint.rglob("*"))
                    if p.is_file()
                },
                "source_files": source_hashes,
                "runner_sha256": digest_file(Path(__file__)),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "settings": {"lr_max_size": 252, "hr_max_size": 252, "compute_dtype": "float16"},
                "seed": 0,
            }
        )
        np.savez_compressed(
            output / "reference.npz", extrinsics=sample.extrinsics_gt, intrinsics=sample.intrinsics
        )
        save()
        torch.manual_seed(0)
        np.random.seed(0)
        model = DAGEAdapter(checkpoint=str(checkpoint / "model.pt"), **manifest["settings"])
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        prediction = model.predict(sample.images, sample.intrinsics)
        torch.cuda.synchronize()
        np.savez_compressed(output / "prediction.npz", extrinsics=prediction.extrinsics)
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds_including_load": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "prediction_file_sha256": digest_file(output / "prediction.npz"),
                "prediction_metadata": prediction.metadata,
            }
        )
        save()
    except Exception:
        manifest.update({"status": "failed", "traceback": traceback.format_exc()})
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.root, args.checkpoint, args.sequence, args.output)
