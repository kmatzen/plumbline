"""Fresh DAGE development run on DAVIS images, without reference camera data.

Validation sequences are deliberately blocked until the evaluation protocol is
frozen. The selected prefix is recorded explicitly; no frame rate is assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def select_frames(inventory: dict, sequence: str, limit: int) -> list[dict]:
    if limit < 2 or limit > 90:
        raise ValueError("development frame limit must be between 2 and 90")
    rows = [s for s in inventory["sequences"] if s["sequence"] == sequence]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise ValueError("select exactly one training sequence; validation is reserved")
    frames = rows[0]["frames"][:limit]
    indices = [f["frame_index"] for f in frames]
    if len(indices) < 2 or indices != sorted(set(indices)):
        raise ValueError("need distinct ordered frame indices")
    return frames


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(root: Path, checkpoint: Path, sequence: str, output: Path, limit: int) -> None:
    if output.exists():
        raise ValueError("preserve prior attempts; choose a new output directory")
    if not checkpoint.is_file():
        raise ValueError("provide a pinned local model.pt checkpoint")
    inventory_path = root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    frames = select_frames(inventory, sequence, limit)
    output.mkdir(parents=True)
    manifest = {
        "status": "preparing",
        "dataset": "davis-2017",
        "sequence": sequence,
        "split": "train",
        "development_run": True,
        "frames": frames,
        "selection": f"first {limit} available frames in recorded order",
        "reference_camera_poses": None,
        "input_intrinsics": None,
        "timestamps": None,
        "inventory_sha256": digest(inventory_path),
        "runner_sha256": digest(Path(__file__)),
        "checkpoint_file": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "settings": {"lr_max_size": 252, "hr_max_size": 252, "compute_dtype": "float16"},
    }

    def save() -> None:
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    try:
        import torch

        from plumbline.models.dage import DAGEAdapter

        images = []
        for frame in frames:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != frame["file_sha256"]:
                raise ValueError("source frame path or hash mismatch")
            with Image.open(path) as image:
                images.append(np.array(image.convert("RGB")))
        batch = np.stack(images)
        package = Path(inspect.getfile(DAGEAdapter)).parents[1]
        manifest.update(
            {
                "status": "running",
                "shape": list(batch.shape),
                "decoded_input_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
                "source_files": {
                    str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))
                },
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "seed": 0,
            }
        )
        save()
        torch.manual_seed(0)
        np.random.seed(0)
        model = DAGEAdapter(checkpoint=str(checkpoint), **manifest["settings"])
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        # Research-only capture: preserve the native normalized calibration that
        # the production pose-only adapter does not currently expose.
        model._load()
        native_infer = model._model.infer
        captured = {}

        def retain_calibration(*args, **kwargs):
            native = native_infer(*args, **kwargs)
            captured["intrinsics_normalized"] = np.asarray(
                native["intrinsics"].detach().float().cpu(), dtype=np.float32
            )
            return native

        model._model.infer = retain_calibration
        prediction = model.predict(batch, intrinsics=None)
        torch.cuda.synchronize()
        calibration = captured["intrinsics_normalized"]
        if calibration.shape != (len(frames), 3, 3) or not np.isfinite(calibration).all():
            raise ValueError("invalid native calibration")
        np.savez_compressed(
            output / "prediction.npz",
            extrinsics=prediction.extrinsics,
            intrinsics_normalized=calibration,
        )
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds_including_load": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "prediction_file_sha256": digest(output / "prediction.npz"),
                "prediction_metadata": prediction.metadata,
                "calibration": {
                    "source": "DAGE predicted; not independently calibrated",
                    "array": "intrinsics_normalized",
                    "convention": "image width/height normalized; principal point 0.5,0.5",
                    "pixel_center_conversion": "u_pixel = width*u_normalized - 0.5; likewise height",
                },
                "functional_verdict": "unreviewed",
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
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    run(args.root, args.checkpoint, args.sequence, args.output, args.limit)
