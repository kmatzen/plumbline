"""Exploratory image-track comparison, separate from the frozen VLM experiment."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from paper.analyze_tapvid_lora import metrics
from paper.evaluate_tapvid_lora import digest, validate_camera_transfer_roster


def predict(candidate_xy, observation):
    """Only candidate geometry and RGB tracking checks enter this decision."""
    if observation is None or observation["xy"] is None:
        return "uncertain", None
    fb, ncc = observation["forward_backward_pixels"], observation["source_patch_ncc"]
    if fb is None or ncc is None or not (math.isfinite(fb) and 0 <= fb <= 1
                                        and math.isfinite(ncc) and .8 <= ncc <= 1):
        return "uncertain", None
    if not all(math.isfinite(v) for v in [*candidate_xy, *observation["xy"]]):
        raise ValueError("nonfinite image coordinates")
    distance = math.dist(candidate_xy, observation["xy"])
    answer = "close" if distance <= 2 else "far" if distance >= 8 else "uncertain"
    return answer, distance


def analyze(corpus, tracker):
    validate_camera_transfer_roster(corpus)
    if (tracker["split"] != "held_out"
            or tracker["source_manifest_sha256"] != corpus["source_manifest_sha256"]
            or set(tracker["sequences"]) != set(corpus["partitions"])):
        raise ValueError("tracker source or video roster mismatch")
    tracks = {}
    for name, entry in tracker["sequences"].items():
        if [t["track"] for t in entry["tracks"]] != list(range(corpus["track_counts"][name])):
            raise ValueError("tracker slot roster mismatch")
        for track in entry["tracks"]:
            records = track.get("records", [])
            by_frame = {r["frame"]:r for r in records}
            if len(by_frame) != len(records):
                raise ValueError("duplicate tracker frame")
            tracks[name, track["track"]] = track, by_frame
    predictions = []
    for row in corpus["records"]:
        answer, distance = "uncertain", None
        if row["geometry_status"] == "renderable":
            track, by_frame = tracks[row["sequence"], row["track"]]
            if (row["source_frame"] != track.get("query_frame")
                    or row["source_xy"] != track.get("query_xy")):
                raise ValueError("source query changed")
            answer, distance = predict(row["candidate_xy"], by_frame.get(row["frame"]))
        predictions.append(dict(id=row["id"], sequence=row["sequence"], model=row["model"],
                                answer=answer, distance_to_rgb_track=distance))
    # Labels enter only after every prediction is fixed.
    indexed = {r["id"]:r for r in corpus["records"]}
    scored = [dict(p, target=indexed[p["id"]]["label"]) for p in predictions
              if indexed[p["id"]]["status"] == "labeled"]
    return dict(scope="exploratory correspondence baseline; not camera quality or DNF",
                policy="existing RGB-only LK observation, FB <=1 pixel and source NCC >=0.8; distance to proposed pixel <=2 close, >=8 far, otherwise uncertain; missing/rejected observations abstain, never infer occlusion",
                timing="added after reference counts and partial base-model responses were observed; not preregistered; no threshold sweep or model selection",
                limitations="shares early image-track evidence with marker fitting; subsequent observations use RGB only; tracking errors can cause incorrect answers; point identity does not establish staticness",
                planned_presentations=len(predictions), predictions=predictions,
                all_planned_answers=dict(Counter(p["answer"] for p in predictions)),
                labeled_metrics=metrics(scored),
                per_video={n:metrics([r for r in scored if r["sequence"] == n]) for n in sorted(corpus["partitions"])},
                by_candidate_method={m:metrics([r for r in scored if r["model"] == m]) for m in ("dage", "da3")})


def run(corpus_path, tracker_path):
    corpus = json.loads(corpus_path.read_text())
    tracker = json.loads(gzip.decompress(tracker_path.read_bytes()))
    # Bind to the exact tracker used by every per-video geometry construction.
    root = corpus_path.parent.parent
    for name, sha in corpus["per_video_corpus_sha256"].items():
        path = root/"corpora"/f"{name}-v1"/"manifest.json"
        if digest(path) != sha or json.loads(path.read_text())["tracker_sha256"] != digest(tracker_path):
            raise ValueError("tracker/corpus provenance mismatch")
    result = analyze(corpus, tracker)
    result["source_sha256"] = {"corpus":digest(corpus_path), "tracker":digest(tracker_path),
                              "analysis":digest(Path(__file__)),
                              "metrics":digest(Path(__file__).with_name("analyze_tapvid_lora.py"))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "tracker", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.corpus, args.tracker)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["labeled_metrics"], indent=2))
