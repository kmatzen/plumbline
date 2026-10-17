"""Freeze camera proposals, then label/render them using reviewed point tracks."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import plumbline.image_evidence as image_geometry
import plumbline.conventions as camera_conventions

from paper.build_tapvid_judge_corpus import PROMPT, render
from paper.run_tapvid_cameras import digest, load_video
from paper.tapvid_camera_markers import LAGS, label_proposals, proposals
from plumbline.conventions import assert_valid_extrinsics


def build(root, sequence, tracker_path, runs, output):
    if output.exists():
        raise ValueError("preserve previous transfer corpus")
    video, source = load_video(root, sequence)
    staging = json.loads((root/"manifest.json").read_text())
    entry = staging["sequences"][sequence]
    tracker = json.loads(gzip.decompress(tracker_path.read_bytes()))
    if tracker["split"] != "held_out" or tracker["source_manifest_sha256"] != source["source_manifest_sha256"]:
        raise ValueError("tracker source mismatch")
    tracking = tracker["sequences"][sequence]
    if tracking["file_sha256"] != source["source_file_sha256"]:
        raise ValueError("tracker video mismatch")
    tracks = tracking["tracks"]
    if [t["track"] for t in tracks] != list(range(entry["points_shape"][0])):
        raise ValueError("all original tracks required")
    size = np.array(video.shape[1:3][::-1])
    records, fits, cameras = [], [], {}
    for method in ("dage", "da3"):
        directory = runs/f"{sequence}-{method}"
        manifest_path = directory/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] not in {"complete", "failed"}:
            raise ValueError("wait for terminal camera attempt")
        if manifest["method"] != method or manifest["sequence"] != sequence:
            raise ValueError("camera identity mismatch")
        cameras[method] = dict(manifest_sha256=digest(manifest_path), status=manifest["status"])
        if manifest["status"] == "complete":
            if any(manifest[k] != source[k] for k in source):
                raise ValueError("camera RGB inputs mismatch")
            if manifest["script_sha256"] != digest(Path(__file__).with_name("run_tapvid_cameras.py")):
                raise ValueError("camera runner changed")
            path = directory/"prediction.npz"
            if digest(path) != manifest["prediction_sha256"]:
                raise ValueError("camera prediction changed")
            with np.load(path, allow_pickle=False) as archive:
                poses, calibration = archive["extrinsics"], archive["intrinsics_normalized"]
            if poses.shape != (len(video), 4, 4) or calibration.shape != (len(video), 3, 3):
                raise ValueError("camera/calibration shape mismatch")
            assert_valid_extrinsics(poses)
            if not np.isfinite(calibration).all():
                raise ValueError("nonfinite calibration")
            cameras[method]["prediction_sha256"] = digest(path)
        for track in tracks:
            if manifest["status"] == "complete":
                fit, proposed = proposals(track, poses, calibration, size)
            else:
                fit = dict(status="camera_run_failed")
                proposed = [dict(lag=lag, frame=None if track.get("query_frame") is None else track["query_frame"]+lag,
                                 status="camera_run_failed", candidate_xy=None) for lag in LAGS]
            fits.append(dict(model=method, track=track["track"], **fit))
            for row in proposed:
                identity = hashlib.sha256(f"camera-transfer-v1:{sequence}:{method}:{track['track']}:{row['lag']}".encode()).hexdigest()[:24]
                records.append(dict(id=identity, sequence=sequence, model=method, track=track["track"],
                                    source_frame=track.get("query_frame"), source_xy=track.get("query_xy"), **row))
    output.mkdir(parents=True)
    # Freeze the complete proposal roster on disk before opening later references.
    geometry = output/"geometry.json"
    geometry.write_text(json.dumps(dict(source=source, tracker_sha256=digest(tracker_path),
                                       cameras=cameras, fits=fits, records=records), indent=2, allow_nan=False)+"\n")
    with np.load(root/entry["file"], allow_pickle=False) as archive:
        points, occ = archive["points"], archive["occluded"]
    if points.shape != tuple(entry["points_shape"]) or occ.shape != points.shape[:2]:
        raise ValueError("reference shape mismatch")
    for track in tracks:
        visible = np.flatnonzero(~occ[track["track"]])
        expected_first = int(visible[0]) if len(visible) else None
        if track.get("query_frame") != expected_first:
            raise ValueError("original source query changed")
        if expected_first is not None and not np.allclose(track["query_xy"], points[track["track"], expected_first]*size, atol=1e-7, rtol=0):
            raise ValueError("original source coordinate changed")
    (output/"images").mkdir()
    labeled = []
    for row in records:
        scored = label_proposals([row], points[row["track"]], occ[row["track"]], size)[0]
        if row["status"] == "renderable":
            try:
                panel = render(video[row["source_frame"]], video[row["frame"]], row["source_xy"], row["candidate_xy"])
            except ValueError:
                scored.update(status="source_display_outside_image", scoring_status="unavailable", label=None)
            else:
                path = output/"images"/f"{row['id']}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("panel write failed")
                scored.update(image=str(path.relative_to(output)), image_sha256=digest(path))
        labeled.append(scored)
    exported = [dict(id=r["id"], image=r["image"], prompt=PROMPT, target=r["label"])
                for r in labeled if r["scoring_status"] == "labeled"]
    export = output/"evaluation.jsonl"
    export.write_text("".join(json.dumps(r)+"\n" for r in exported))
    result = dict(scope="camera-projected correspondence validation; no staticness or camera-DNF labels",
                  sequence=sequence, prompt=PROMPT, planned_presentations=len(records),
                  geometry_sha256=digest(geometry), source_manifest_sha256=source["source_manifest_sha256"],
                  source_file_sha256=source["source_file_sha256"], tracker_sha256=digest(tracker_path),
                  script_sha256=digest(Path(__file__)), export_sha256=digest(export),
                  geometry_dependencies_sha256={"plumbline.image_evidence":digest(Path(image_geometry.__file__)),
                                                "plumbline.conventions":digest(Path(camera_conventions.__file__))},
                  dependencies_sha256={n:digest(Path(__file__).with_name(n)) for n in
                      ("tapvid_camera_markers.py", "build_anchor_preview.py", "build_tapvid_judge_corpus.py", "run_tapvid_cameras.py")},
                  counts=dict(Counter(r["label"] or r["scoring_status"] if r["status"] == "renderable" else r["status"] for r in labeled)),
                  records=labeled)
    (output/"manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sequence")
    for name in ("tracker", "runs", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(build(args.root, args.sequence, args.tracker, args.runs, args.output)["counts"])
