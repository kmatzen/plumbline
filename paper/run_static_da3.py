"""DA3 on the fixed static-control RGB sample, with the prior query-chunk setting."""

import argparse
import hashlib
import inspect
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np

if __package__:
    from .run_static_dage import digest, load_rgb
    from .query_chunk_attention import query_chunk_attention
else:
    from run_static_dage import digest, load_rgb
    from query_chunk_attention import query_chunk_attention


def run(root, checkpoint, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    if not all((checkpoint / name).is_file() for name in ("model.safetensors", "config.json")):
        raise ValueError("pinned local DA3 snapshot required")
    import torch
    from plumbline.models.depth_anything_3 import DepthAnything3Adapter

    if not torch.cuda.is_available():
        raise ValueError("run in the CUDA environment")
    output.mkdir(parents=True)
    record = {"status": "preparing", "scope": "fixed RGB development prediction; not DNF validation",
              "model": "DA3-LARGE-1.1"}

    def save():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    try:
        save()
        images, frames, source_hash = load_rgb(root)
        package = Path(inspect.getfile(DepthAnything3Adapter)).parents[1]
        record.update(status="running", source_manifest_sha256=source_hash, frames=frames,
                      runner_sha256=digest(Path(__file__)), seed=0,
                      input_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                      checkpoint=str(checkpoint),
                      checkpoint_files={p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()},
                      source_files={str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
                      input_loader_sha256=digest(Path(__file__).with_name("run_static_dage.py")),
                      query_chunk_script_sha256=digest(Path(__file__).with_name("query_chunk_attention.py")),
                      settings={"process_res": 252, "process_res_method": "upper_bound_resize", "query_chunk": 256},
                      python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), inputs="RGB only; no depth, intrinsics, or reference poses passed to model",
                      attention_caveat="query-chunk kernel tested; end-to-end equivalence with unchunked inference not established")
        save()
        torch.manual_seed(0)
        np.random.seed(0)
        model = DepthAnything3Adapter(checkpoint=str(checkpoint), process_res=252)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with query_chunk_attention(256):
            prediction = model.predict(images)
        torch.cuda.synchronize()
        np.savez_compressed(output / "prediction.npz", extrinsics=prediction.extrinsics,
                            intrinsics_processed_pixels=prediction.intrinsics)
        record.update(status="complete", prediction_file_sha256=digest(output / "prediction.npz"),
                      elapsed_seconds_including_load=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), prediction_metadata=prediction.metadata,
                      processed_shape_hw=list(prediction.depth.shape[1:]),
                      calibration="saved intrinsics are in processed-image pixel coordinates; primary comparison uses common published calibration")
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
