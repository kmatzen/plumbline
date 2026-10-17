"""Frozen correspondence test inputs from the 20 previously reserved videos.

These videos already have reported classical-baseline results. They are not a
new untouched benchmark; they are disjoint from the learned judge's development
videos. This script performs no model inference or threshold selection.
"""

import argparse
import gzip
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from paper.build_tapvid_judge_corpus import PROMPT, digest, render, target_label


def test_names(staging, predictions):
    names = sorted(n for n, entry in staging["sequences"].items() if entry["split"] == "held_out")
    if len(names) != 20 or set(predictions["sequences"]) != set(names):
        raise ValueError("exactly the twenty held-out videos required")
    return names


def run(root, source, output):
    if output.exists():
        raise ValueError("preserve previous corpus")
    staging = json.loads((root/"manifest.json").read_text())
    predictions = json.loads(gzip.decompress(source.read_bytes()))
    if predictions["source_manifest_sha256"] != digest(root/"manifest.json"):
        raise ValueError("source mismatch")
    names = test_names(staging, predictions)
    output.mkdir(parents=True)
    (output/"images").mkdir()
    records = []
    cv2.setNumThreads(1)
    for name in names:
        entry = staging["sequences"][name]
        path = root/entry["file"]
        if digest(path) != entry["file_sha256"]:
            raise ValueError("changed video or annotations")
        with np.load(path, allow_pickle=False) as archive:
            video, points, occ = archive["video"], archive["points"], archive["occluded"]
        h, w = video.shape[1:3]
        for track in predictions["sequences"][name]["tracks"]:
            first = track.get("query_frame")
            candidates = {r["frame"]: r for r in track.get("records", [])}
            for lag in (5, 15, 30, 60):
                frame = None if first is None else first+lag
                identity = hashlib.sha256(f'{name}:{track["track"]}:{frame if frame is not None else "noquery"+str(lag)}'.encode()).hexdigest()[:20]
                record = {"id": identity, "sequence": name, "partition": "test", "track": track["track"],
                          "query_frame": first, "frame": frame, "lag": lag, "label": None}
                row = candidates.get(frame)
                if first is None:
                    record["status"] = "no_visible_query"
                elif frame >= len(video):
                    record["status"] = "beyond_video"
                elif row is None or row["xy"] is None:
                    record["status"] = "candidate_unavailable"
                else:
                    try:
                        panel = render(video[first], video[frame], track["query_xy"], row["xy"])
                    except ValueError:
                        record["status"] = "rounded_display_point_outside"
                        records.append(record)
                        continue
                    label = target_label(row["xy"], points[track["track"], frame]*[w, h], bool(occ[track["track"], frame]))
                    image = output/"images"/(identity+".png")
                    if not cv2.imwrite(str(image), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("image write failed")
                    record.update(status="labeled" if label else "borderline_unlabeled", label=label,
                                  image=str(image.relative_to(output)), image_sha256=digest(image),
                                  source_xy=track["query_xy"], candidate_xy=row["xy"])
                records.append(record)
        print("prepared", name, flush=True)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate example identity")
    examples = [{"id": r["id"], "image": r["image"], "prompt": PROMPT, "target": r["label"]}
                for r in records if r["status"] == "labeled"]
    export = output/"test.jsonl"
    export.write_text("".join(json.dumps(row)+"\n" for row in examples))
    manifest = {"scope": "correspondence-only test preparation; no staticness, camera, or DNF labels",
                "history": "twenty videos already evaluated for classical tracking; no learned-judge results used here; separate from all ten learned-judge development videos",
                "selection": "same fixed lags 5/15/30/60 and every first-visible track; all missing and borderline cases retained",
                "label_policy": "same reviewed annotations and native-pixel thresholds as development: close<=2, far>=8, occluded separate, intermediate unlabeled",
                "source_manifest_sha256": digest(root/"manifest.json"), "predictions_sha256": digest(source),
                "script_sha256": digest(Path(__file__)),
                "renderer_and_label_helper_sha256": digest(Path(__file__).with_name("build_tapvid_judge_corpus.py")),
                "opencv_version": cv2.__version__, "prompt": PROMPT,
                "partitions": {name:"test" for name in names}, "exports_sha256": {"test":digest(export)},
                "records": records}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print("prepared",len(records),"planned examples and",len(examples),"labeled examples; no inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root,args.source,args.output)
