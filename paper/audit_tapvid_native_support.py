"""Secondary native-pixel and whole-track audit of frozen image-check decisions."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def accepted(row):
    # Exact frozen primary decision; annotation data never enters this rule.
    return (row["xy"] is not None and row["forward_backward_pixels"] is not None
            and row["forward_backward_pixels"] <= 1 and row["source_patch_ncc"] is not None
            and row["source_patch_ncc"] >= .8)


def audit_track(track, points, occluded, width, height):
    observed = []
    for row in track["records"]:
        if not accepted(row):
            continue
        frame = row["frame"]
        residual = np.array(row["xy"]) - points[frame] * [width, height]
        observed.append({"frame": frame, "lag": frame-track["query_frame"],
                         "visible": not bool(occluded[frame]),
                         "absolute_xy_error_native": np.abs(residual).tolist()})
    thresholds = {}
    for tolerance in (1, 2, 4, 8):
        correct = sum(r["visible"] and max(r["absolute_xy_error_native"]) <= tolerance for r in observed)
        thresholds[str(tolerance)] = {
            "accepted_in_native_box": correct,
            "all_accepted_in_native_box": bool(correct == len(observed)) if observed else None,
            "all_accepted_in_native_box_with_two_or_more": bool(correct == len(observed)) if len(observed) >= 2 else None}
    return {"track": track["track"], "planned_later_observations": len(track["records"]),
            "accepted_count": len(observed), "accepted_occluded": sum(not r["visible"] for r in observed),
            "accepted": observed, "native_box_tolerances": thresholds}


def summarize(tracks):
    n = sum(t["accepted_count"] for t in tracks)
    eligible = [t for t in tracks if t["accepted_count"] >= 2]
    result = {"tracks": len(tracks), "zero_accepted_tracks": sum(t["accepted_count"] == 0 for t in tracks),
              "one_accepted_tracks": sum(t["accepted_count"] == 1 for t in tracks),
              "at_least_two_accepted_tracks": len(eligible), "accepted_observations": n,
              "accepted_occluded": sum(t["accepted_occluded"] for t in tracks), "tolerances": {}}
    for tol in ("1", "2", "4", "8"):
        good = sum(t["native_box_tolerances"][tol]["accepted_in_native_box"] for t in tracks)
        clean = sum(t["native_box_tolerances"][tol]["all_accepted_in_native_box_with_two_or_more"] for t in eligible)
        result["tolerances"][tol] = {"accepted_in_box": good, "accepted_in_box_fraction": good/n if n else None,
                                      "whole_tracks_in_box": clean,
                                      "whole_track_fraction_among_two_or_more": clean/len(eligible) if eligible else None}
    return result


def run(root, source, split, output):
    if output.exists():
        raise ValueError("preserve previous audit")
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads(gzip.decompress(source.read_bytes()))
    if result["source_manifest_sha256"] != digest(root / "manifest.json"):
        raise ValueError("annotation source mismatch")
    expected = {n for n, e in manifest["sequences"].items() if e["split"] == split}
    if set(result["sequences"]) != expected:
        raise ValueError("split membership mismatch")
    sequences = {}
    for name, record in result["sequences"].items():
        entry = manifest["sequences"][name]
        path = root / entry["file"]
        if digest(path) != entry["file_sha256"] or record["file_sha256"] != entry["file_sha256"]:
            raise ValueError("changed source arrays")
        with np.load(path, allow_pickle=False) as a:
            points, occ = a["points"], a["occluded"]
        height, width = entry["video_shape"][1:3]
        if [t["track"] for t in record["tracks"]] != list(range(len(points))):
            raise ValueError("track slots changed")
        tracks = [audit_track(t, points[t["track"]], occ[t["track"]], width, height) for t in record["tracks"]]
        if sum(t["accepted_count"] for t in tracks) != record["counts"]["both"]["accepted"]:
            raise ValueError("frozen primary decisions changed")
        sequences[name] = {"tracks": tracks, "summary": summarize(tracks)}
    report = {"scope": "secondary descriptive audit after primary held-out results; no threshold tuning or new acceptance rule",
              "split": split, "source_sha256": digest(source), "source_manifest_sha256": digest(root / "manifest.json"),
              "script_sha256": digest(Path(__file__)),
              "policy": "visible and both native coordinate residuals <= tolerance; occluded is not correct; whole-track denominator requires at least two accepted later observations; zero/one accepted retained separately",
              "limitations": "reviewed annotations have uncertainty; spatial agreement does not establish staticness; selected observations may be adjacent and provide little parallax; no camera estimates evaluated",
              "sequences": sequences}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--split", choices=("development", "held_out"), required=True)
    args = parser.parse_args()
    result = run(args.root, args.source, args.split, args.output)
    for name, row in result["sequences"].items():
        print(name, row["summary"])
