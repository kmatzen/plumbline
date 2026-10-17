"""Fresh DAGE inference on the fixed static-control RGB sample; no reference poses."""

import argparse
import hashlib
import inspect
import json
import platform
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


def load_rgb(root):
    manifest_path = root / "staging-manifest.json"
    source = json.loads(manifest_path.read_text())
    if len(source["frames"]) != 90:
        raise ValueError("expected fixed 90-frame sample")
    images, frames = [], []
    for index, frame in enumerate(source["frames"]):
        path = (root / frame["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != frame["file_sha256"]:
            raise ValueError("RGB path/hash mismatch")
        if frame["frame"] != index or frame["rgb_index"] != index * 3:
            raise ValueError("sample order mismatch")
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        if rgb.shape != (480, 640, 3):
            raise ValueError("expected native image size")
        images.append(rgb)
        frames.append({key: frame[key] for key in ("frame", "rgb_index", "path", "timestamp", "file_sha256")})
        frames[-1]["decoded_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    return np.stack(images), frames, digest(manifest_path)


def run(root, checkpoint, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    if not (checkpoint / "model.pt").is_file():
        raise ValueError("pinned DAGE snapshot required")
    import torch
    from plumbline.models.dage import DAGEAdapter

    if not torch.cuda.is_available():
        raise ValueError("run in the CUDA environment")
    output.mkdir(parents=True)
    record = {"status": "preparing", "scope": "fixed RGB development prediction; not DNF validation"}

    def save():
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    try:
        save()
        images, frames, source_hash = load_rgb(root)
        package = Path(inspect.getfile(DAGEAdapter)).parents[1]
        settings = {"lr_max_size": 252, "hr_max_size": 252, "compute_dtype": "float16"}
        record.update(status="running", source_manifest_sha256=source_hash, frames=frames,
                      runner_sha256=digest(Path(__file__)), settings=settings, seed=0,
                      input_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                      checkpoint=str(checkpoint),
                      checkpoint_files={str(p.relative_to(checkpoint)): digest(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
                      source_files={str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
                      python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),
                      inputs="RGB plus published Freiburg-3 calibration; no depth or reference poses read for inference")
        calibration = np.repeat(np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])[None], 90, axis=0)
        save()
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model = DAGEAdapter(checkpoint=str(checkpoint / "model.pt"), **settings)
        prediction = model.predict(images, calibration)
        torch.cuda.synchronize()
        np.savez_compressed(output / "prediction.npz", extrinsics=prediction.extrinsics)
        record.update(status="complete", prediction_file_sha256=digest(output / "prediction.npz"),
                      elapsed_seconds_including_load=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), prediction_metadata=prediction.metadata)
        save()
    except Exception:
        record.update(status="failed", traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.checkpoint, args.output)
