"""Unlabeled camera-projected marker panels for a frozen correspondence judge."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.build_anchor_preview import fit_point
from paper.build_letter_camera_review import project_marker
from paper.build_tapvid_judge_corpus import PROMPT, render
from paper.probe_real_evidence_sensitivity import K
from paper.probe_static_point_support import load_prediction
from paper.run_static_dage import load_rgb
from paper.stage_static_rgbd_control import digest

FIT = list(range(1, 11))
QUERIES = (30, 60, 89)


def fit_marker(track, poses):
    """No later image observations or reference poses are consulted."""
    if any(track[i]["xy"] is None for i in FIT):
        return dict(status="missing_prefix_observations")
    size = np.array([640, 480])
    normalized = K.copy()
    normalized[:2, 2] += .5
    normalized[:2] /= size[:, None]
    xy = np.array([track[i]["xy"] for i in FIT])
    point = fit_point(xy, poses[FIT], np.repeat(normalized[None], len(FIT), axis=0), size)
    if point is None or not np.isfinite(point).all():
        return dict(status="degenerate_prefix_fit")
    projected = [project_marker(point, poses[i]) for i in FIT]
    if any(p is None for p in projected):
        return dict(status="nonpositive_prefix_depth", point_world=point.tolist())
    return dict(status="fitted", point_world=point.tolist(), source_xy=track[1]["xy"],
                prefix_max_pixel_error=float(np.linalg.norm(np.array(projected)-xy, axis=1).max()))


def query_marker(fit, pose):
    if fit["status"] != "fitted":
        return dict(status=fit["status"])
    xy = project_marker(np.asarray(fit["point_world"]), pose)
    if xy is None:
        return dict(status="nonpositive_query_depth")
    if not np.isfinite(xy).all():
        return dict(status="nonfinite_query_projection")
    rounded = np.rint(xy)
    if not (0 <= rounded[0] < 640 and 0 <= rounded[1] < 480):
        return dict(status="query_outside_image", candidate_xy=xy.tolist())
    return dict(status="renderable", candidate_xy=xy.tolist())


def build(root, tracks_path, predictions, output):
    if output.exists():
        raise ValueError("preserve previous corpus")
    images, frames, source_hash = load_rgb(root)
    control = json.loads(tracks_path.read_text())
    if control["source_manifest_sha256"] != source_hash or len(control["tracks"]) != 12:
        raise ValueError("expected same source and all twelve tracks")
    if any(len(t) != 90 for t in control["tracks"]):
        raise ValueError("expected all ninety track records")
    output.mkdir(parents=True)
    (output/"images").mkdir()
    records, fits, provenance = [], [], {}
    image_hash = hashlib.sha256(images.tobytes()).hexdigest()
    for model in ("dage", "da3"):
        poses, provenance[model] = load_prediction(predictions/f"{model}-v1", source_hash, frames, image_hash)
        for slot, track in enumerate(control["tracks"]):
            fitted = fit_marker(track, poses)
            fits.append(dict(model=model, slot=slot, **fitted))
            for query in QUERIES:
                identity = hashlib.sha256(f"{source_hash}:{model}:{slot}:{query}".encode()).hexdigest()[:24]
                row = dict(id=identity, model=model, slot=slot, frame=query,
                           **query_marker(fitted, poses[query]))
                if row["status"] == "renderable":
                    try:
                        panel = render(images[1], images[query], fitted["source_xy"], row["candidate_xy"])
                    except ValueError:
                        row["status"] = "source_display_outside_image"
                    else:
                        path = output/"images"/f"{identity}.png"
                        if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                            raise RuntimeError("panel write failed")
                        row.update(image=str(path.relative_to(output)), image_sha256=digest(path))
                records.append(row)
    examples = [dict(id=r["id"], image=r["image"], prompt=PROMPT) for r in records if r["status"] == "renderable"]
    export = output/"inference.jsonl"
    export.write_text("".join(json.dumps(r)+"\n" for r in examples))
    dependencies = [Path(__file__), Path(__file__).with_name("build_anchor_preview.py"),
                    Path(__file__).with_name("build_letter_camera_review.py"),
                    Path(__file__).with_name("build_tapvid_judge_corpus.py"),
                    Path(__file__).with_name("probe_real_evidence_sensitivity.py"),
                    Path(__file__).with_name("probe_static_point_support.py"),
                    Path(__file__).with_name("run_static_dage.py")]
    result = dict(scope="unlabeled development camera-marker judgments; no camera accuracy or DNF labels",
                  policy="all twelve slots, both methods, fixed fit frames 1..10 and queries 30/60/89; no later tracked positions used; no point replacement or query selection by residual; same frozen correspondence prompt and renderer",
                  limitations="a single fitted point is not a geometric certainty; image identity/staticness, calibration, prefix fits and cameras can cause errors; models used all frames; correspondence judgments cannot establish camera DNF; invalid projections are not failure labels",
                  source_manifest_sha256=source_hash, tracks_sha256=digest(tracks_path),
                  provenance=provenance, calibration=K.tolist(), source_frame=1, fit_frames=FIT,
                  query_frames=list(QUERIES), prompt=PROMPT, planned_presentations=72,
                  export_sha256=digest(export), opencv_version=cv2.__version__,
                  dependencies_sha256={p.name:digest(p) for p in dependencies}, fits=fits,
                  counts=dict(Counter(r["status"] for r in records)), records=records)
    (output/"manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tracks", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(build(args.root, args.tracks, args.predictions, args.output)["counts"])
