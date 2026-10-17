"""Decode all four fixed pilot scenes; separate RGB from scoring references."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def images(values, shape, dtype, rgb=False):
    frames = []
    for value in values:
        frame = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError("invalid PNG")
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    array = np.stack(frames)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError("decoded image shape or dtype mismatch")
    return array


def decode(source, output):
    from tfrecord import tfrecord_iterator, example_pb2
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("complete staged input required")
    for entry in manifest["files"]:
        if digest(source/entry["file"]) != entry["sha256"]:
            raise ValueError("staged file changed")
    info = json.loads((source/"dataset_info.json").read_text())
    split = next(s for s in info["splits"] if s["name"] == "validation")
    expected = int(split["shardLengths"][0])
    shard = source/"movi_e-validation.tfrecord-00000-of-00064"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="decoding", scope="geometry development only; no validated projection or DNF labels",
                  source_manifest_sha256=digest(source/"manifest.json"), shard_sha256=digest(shard),
                  expected_scenes=expected, scenes=[], script_sha256=digest(Path(__file__)),
                  versions={n:importlib.metadata.version(n) for n in ("tfrecord","protobuf","crc32c","numpy")})
    try:
        for index, payload in enumerate(tfrecord_iterator(str(shard))):
            example = example_pb2.Example()
            example.ParseFromString(payload)
            features = example.features.feature

            def values(key, kind):
                if key not in features or features[key].WhichOneof("kind") != kind:
                    raise ValueError(f"missing or wrong feature type: {key}")
                return getattr(features[key], kind).value

            def scalar(key):
                v = values(key, "int64_list")
                if len(v) != 1:
                    raise ValueError("nonscalar metadata")
                return int(v[0])

            t, h, w, n = [scalar("metadata/"+k) for k in ("num_frames","height","width","num_instances")]
            if (t,h,w) != (24,256,256) or n <= 0:
                raise ValueError("unexpected pilot scene dimensions")
            name = values("metadata/video_name", "bytes_list")[0].decode()
            if name in {s["name"] for s in result["scenes"]}:
                raise ValueError("duplicate scene")
            rgb = images(values("video","bytes_list"), (t,h,w,3), np.uint8, rgb=True)
            references = {}
            for key, shape in (("camera/positions",(t,3)),("camera/quaternions",(t,4)),
                               ("instances/positions",(n,t,3)),("instances/quaternions",(n,t,4)),
                               ("instances/image_positions",(n,t,2)),("instances/bboxes_3d",(n,t,8,3)),
                               ("camera/focal_length",(1,)),("camera/sensor_width",(1,)),
                               ("camera/field_of_view",(1,)),("metadata/depth_range",(2,))):
                array = np.asarray(values(key,"float_list"),dtype=np.float32).reshape(shape)
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite geometry")
                references[key.replace("/","_")] = array
            references["initial_is_dynamic"] = np.asarray(values("instances/is_dynamic","int64_list"),dtype=np.int64).reshape(n)
            for key,dtype,shape,rgb_order in (("depth",np.uint16,(t,h,w),False),
                                            ("segmentations",np.uint8,(t,h,w),False),
                                            ("object_coordinates",np.uint16,(t,h,w,3),True)):
                references[key] = images(values(key,"bytes_list"),shape,dtype,rgb=rgb_order)
            rgb_path, ref_path = output/f"scene-{index:02d}-rgb.npz", output/f"scene-{index:02d}-references.npz"
            np.savez_compressed(rgb_path, video=rgb)
            np.savez_compressed(ref_path, **references)
            result["scenes"].append(dict(index=index,name=name,frames=t,instances=n,
                rgb_file=rgb_path.name,rgb_sha256=digest(rgb_path),decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                reference_file=ref_path.name,reference_sha256=digest(ref_path),
                record_sha256=hashlib.sha256(payload).hexdigest()))
            print(name, t, n, flush=True)
        if len(result["scenes"]) != expected:
            raise ValueError("shard scene count mismatch")
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed",error=repr(exc))
        raise
    finally:
        (output/"manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    decode(args.source,args.output)
