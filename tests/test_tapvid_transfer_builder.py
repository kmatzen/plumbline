import gzip
import json
from pathlib import Path

import numpy as np

from paper.build_tapvid_camera_transfer import build
from paper.run_tapvid_cameras import digest, load_video
from paper.tapvid_camera_markers import project


def test_geometry_frozen_before_label_export_and_all_queries_retained(tmp_path):
    root = tmp_path/"source"
    root.mkdir()
    size = np.array([80, 64])
    video = np.zeros((20, 64, 80, 3), np.uint8)
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    poses[:, 0, 3] = np.linspace(0, .1, 20)
    k = np.repeat(np.array([[.8, 0, .5], [0, 1, .5], [0, 0, 1.]])[None], 20, axis=0)
    xy = np.array([project([0, 0, 4], p, ki, size) for p, ki in zip(poses, k)])
    points = (xy/size)[None]
    source = root/"sample.npz"
    np.savez(source, video=video, points=points, occluded=np.zeros((1, 20), bool))
    import hashlib
    entry = dict(split="held_out", file=source.name, file_sha256=digest(source),
                 video_shape=list(video.shape), points_shape=list(points.shape),
                 decoded_array_sha256=dict(video=hashlib.sha256(video.tobytes()).hexdigest()))
    (root/"manifest.json").write_text(json.dumps(dict(sequences=dict(sample=entry))))
    _, provenance = load_video(root, "sample")
    track = dict(track=0, query_frame=0, query_xy=xy[0].tolist(),
                 records=[dict(frame=i, xy=xy[i].tolist()) for i in range(1, 20)])
    tracker = tmp_path/"tracks.gz"
    tracker.write_bytes(gzip.compress(json.dumps(dict(split="held_out",
        source_manifest_sha256=provenance["source_manifest_sha256"],
        sequences=dict(sample=dict(file_sha256=digest(source), tracks=[track])))).encode()))
    runs = tmp_path/"runs"
    for method in ("dage", "da3"):
        directory = runs/f"sample-{method}"
        directory.mkdir(parents=True)
        prediction = directory/"prediction.npz"
        np.savez(prediction, extrinsics=poses, intrinsics_normalized=k)
        manifest = dict(status="complete", method=method, sequence="sample", **provenance,
                        prediction_sha256=digest(prediction),
                        script_sha256=digest(Path(__file__).parents[1]/"paper/run_tapvid_cameras.py"))
        (directory/"manifest.json").write_text(json.dumps(manifest))
    output = tmp_path/"corpus"
    result = build(root, "sample", tracker, runs, output)
    assert result["planned_presentations"] == 6
    assert result["counts"] == {"close": 2, "query_beyond_video": 4}
    geometry = json.loads((output/"geometry.json").read_text())
    assert all("label" not in r for r in geometry["records"])
    exports = [json.loads(line) for line in (output/"evaluation.jsonl").read_text().splitlines()]
    assert len(exports) == 2 and all(r["target"] == "close" for r in exports)
