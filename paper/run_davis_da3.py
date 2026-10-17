"""DA3 development run on exactly the frames of a completed DAVIS training run."""

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


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_input_intrinsics(k, processed_hw, input_hw):
    """Invert DA3's scale-only resize policy, then encode integer-center input K.

    Applies only to same-sized inputs with upper_bound_resize (no cropping).
    This follows DA3 InputProcessor._resize_ixt rather than imposing a different
    half-pixel resize convention. The final +0.5 is our normalized UV encoding.
    """
    matrix = np.array(k, dtype=np.float64, copy=True)
    ph, pw = processed_hw
    ih, iw = input_hw
    if min(ph, pw, ih, iw) <= 0 or matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError("valid image shapes and batched intrinsics required")
    matrix[:, 0, :] *= iw / pw
    matrix[:, 1, :] *= ih / ph
    matrix[:, 0, 2] += 0.5
    matrix[:, 1, 2] += 0.5
    matrix[:, 0, :] /= iw
    matrix[:, 1, :] /= ih
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite calibration")
    return matrix.astype(np.float32)


def run(source: Path, root: Path, checkpoint: Path, output: Path, query_chunk: int = 0):
    if output.exists():
        raise ValueError("preserve previous attempts")
    original = json.loads(source.read_text())
    if original["status"] != "complete" or original["split"] != "train":
        raise ValueError("completed training source required")
    if not all((checkpoint / name).is_file() for name in ("model.safetensors", "config.json")):
        raise ValueError("pinned local checkpoint required")
    output.mkdir(parents=True)
    manifest = {
        "status": "preparing",
        "dataset": "davis-2017",
        "split": "train",
        "development_run": True,
        "sequence": original["sequence"],
        "frames": original["frames"],
        "paired_source_manifest_sha256": digest(source),
        "reference_camera_poses": None,
        "input_intrinsics": None,
        "timestamps": None,
        "runner_sha256": digest(Path(__file__)),
        "model": "DA3-LARGE-1.1",
        "checkpoint_files": {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()},
        "settings": {"process_res": 252, "process_res_method": "upper_bound_resize"},
    }

    def save():
        temp = output / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(output / "manifest.json")

    save()
    try:
        from contextlib import nullcontext

        import torch

        from plumbline.models.depth_anything_3 import DepthAnything3Adapter

        images = []
        for frame in manifest["frames"]:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != frame["file_sha256"]:
                raise ValueError("source image mismatch")
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        batch = np.stack(images)
        package = Path(inspect.getfile(DepthAnything3Adapter)).parents[1]
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
        model = DepthAnything3Adapter(checkpoint=str(checkpoint), process_res=252)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        if query_chunk:
            from query_chunk_attention import query_chunk_attention

            manifest["settings"]["query_chunk"] = query_chunk
            manifest["query_chunk_script_sha256"] = digest(
                Path(__file__).with_name("query_chunk_attention.py")
            )
            save()
        with query_chunk_attention(query_chunk) if query_chunk else nullcontext():
            prediction = model.predict(batch)
        torch.cuda.synchronize()
        normalized = normalized_input_intrinsics(
            prediction.intrinsics, prediction.depth.shape[1:], batch.shape[1:3]
        )
        np.savez_compressed(
            output / "prediction.npz",
            extrinsics=prediction.extrinsics,
            intrinsics_normalized=normalized,
            intrinsics_processed_pixels=prediction.intrinsics,
        )
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds_including_load": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "prediction_file_sha256": digest(output / "prediction.npz"),
                "prediction_metadata": prediction.metadata,
                "functional_verdict": "unreviewed",
                "processed_shape_hw": list(prediction.depth.shape[1:]),
                "calibration": {
                    "array": "intrinsics_normalized",
                    "source": "DA3 estimated; not independent calibration",
                    "conversion": "inverse scale-only resize, then (pixel+0.5)/input_size",
                },
            }
        )
        save()
    except Exception:
        manifest.update({"status": "failed", "traceback": traceback.format_exc()})
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "root", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--query-chunk", type=int, default=0)
    args = parser.parse_args()
    run(args.source, args.root, args.checkpoint, args.output, args.query_chunk)
