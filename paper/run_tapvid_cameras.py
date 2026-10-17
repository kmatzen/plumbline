"""RGB-only camera predictions on the existing twenty TAP-Vid test videos."""

import argparse
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path

import numpy as np

if __package__:
    from .query_chunk_attention import query_chunk_attention
    from .run_davis_da3 import normalized_input_intrinsics
else:
    from query_chunk_attention import query_chunk_attention
    from run_davis_da3 import normalized_input_intrinsics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_video(root, sequence):
    source = root/"manifest.json"
    manifest = json.loads(source.read_text())
    entry = manifest["sequences"][sequence]
    if entry["split"] != "held_out":
        raise ValueError("only existing held-out videos allowed; no calibration videos")
    path = (root/entry["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != entry["file_sha256"]:
        raise ValueError("source video path or hash mismatch")
    # Do not read points or occlusion arrays: those belong to later validation.
    with np.load(path, allow_pickle=False) as archive:
        video = archive["video"]
    sha = hashlib.sha256(video.tobytes()).hexdigest()
    if (list(video.shape) != entry["video_shape"] or video.dtype != np.uint8
            or video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0
            or sha != entry["decoded_array_sha256"]["video"]):
        raise ValueError("decoded RGB mismatch")
    return video, dict(source_manifest_sha256=digest(source), source_file_sha256=digest(path),
                      decoded_rgb_sha256=sha, shape=list(video.shape),
                      frame_sha256=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in video])


def run(root, sequence, method, checkpoint, output):
    if output.exists():
        raise ValueError("preserve previous attempts")
    if method not in {"dage", "da3"}:
        raise ValueError("unknown method")
    required = ("model.pt",) if method == "dage" else ("config.json", "model.safetensors")
    if not all((checkpoint/name).is_file() for name in required):
        raise ValueError("pinned local checkpoint required")
    output.mkdir(parents=True)
    report = dict(status="preparing", sequence=sequence, method=method, seed=0,
                  scope="camera outputs for correspondence-transfer validation, not camera DNF labels",
                  input_policy="all original RGB frames only; no points, occlusion annotations, measured calibration, or reference camera poses",
                  script_sha256=digest(Path(__file__)))

    def save():
        temporary = output/"manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"manifest.json")

    save()
    try:
        import torch
        from plumbline.models.dage import DAGEAdapter
        from plumbline.models.depth_anything_3 import DepthAnything3Adapter
        from plumbline.conventions import assert_valid_extrinsics
        video, provenance = load_video(root, sequence)
        package = Path(inspect.getfile(DAGEAdapter)).parents[1]
        report.update(status="running", **provenance, checkpoint=str(checkpoint),
                      checkpoint_files={name:digest(checkpoint/name) for name in required},
                      source_files={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob("*.py"))},
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                      helpers_sha256={name:digest(Path(__file__).with_name(name)) for name in
                                      ("run_davis_da3.py", "query_chunk_attention.py")})
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if method == "dage":
            settings = dict(lr_max_size=252, hr_max_size=252, compute_dtype="float16")
            report["settings"] = settings
            save()
            model = DAGEAdapter(checkpoint=str(checkpoint/"model.pt"), **settings)
            model._load()
            native_infer = model._model.infer
            captured = {}

            def retain(*args, **kwargs):
                native = native_infer(*args, **kwargs)
                captured["k"] = np.asarray(native["intrinsics"].detach().float().cpu(), dtype=np.float32)
                return native

            model._model.infer = retain
            prediction = model.predict(video, intrinsics=None)
            calibration = captured["k"]
        else:
            report["settings"] = dict(process_res=252, process_res_method="upper_bound_resize", query_chunk=256)
            report["attention_caveat"] = "end-to-end equivalence to unchunked inference not established"
            save()
            model = DepthAnything3Adapter(checkpoint=str(checkpoint), process_res=252)
            with query_chunk_attention(256):
                prediction = model.predict(video)
            calibration = normalized_input_intrinsics(prediction.intrinsics, prediction.depth.shape[1:], video.shape[1:3])
        torch.cuda.synchronize()
        if calibration.shape != (len(video), 3, 3) or not np.isfinite(calibration).all():
            raise ValueError("invalid predicted calibration")
        if prediction.extrinsics.shape != (len(video), 4, 4):
            raise ValueError("invalid camera frame count")
        assert_valid_extrinsics(prediction.extrinsics)
        np.savez_compressed(output/"prediction.npz", extrinsics=prediction.extrinsics,
                            intrinsics_normalized=calibration)
        report.update(status="complete", prediction_sha256=digest(output/"prediction.npz"),
                      prediction_metadata=prediction.metadata,
                      calibration="predicted width/height-normalized intrinsics; raster coordinates = normalized*size - 0.5; not reference calibration",
                      elapsed_seconds_including_load=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("method", choices=("dage", "da3"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.root, args.sequence, args.method, args.checkpoint, args.output)
