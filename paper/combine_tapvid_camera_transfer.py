"""Combine the full fixed transfer roster, with an image-only patch baseline."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

from paper.build_tapvid_judge_corpus import PROMPT
from paper.evaluate_tapvid_lora import validate_camera_transfer_roster
from paper.probe_tapvid_identity import correlation, patch
from paper.run_tapvid_cameras import digest, load_video


def combine(root, corpora, plan_path, output):
    if output.exists():
        raise ValueError("preserve previous combined corpus")
    staging = json.loads((root/"manifest.json").read_text())
    plan = json.loads(plan_path.read_text())
    names = sorted(n for n, e in staging["sequences"].items() if e["split"] == "held_out")
    if len(names) != 20 or names != plan["sequences"] or plan["methods"] != ["dage", "da3"]:
        raise ValueError("fixed twenty-video/two-method protocol required")
    # Do not create a partial test corpus while any planned video is missing.
    manifests = {n:json.loads((corpora/f"{n}-v1"/"manifest.json").read_text()) for n in names}
    all_records, input_hashes = [], {}
    for name in names:
        path = corpora/f"{name}-v1"
        manifest = manifests[name]
        if (manifest["sequence"] != name or manifest["prompt"] != PROMPT
                or manifest["source_manifest_sha256"] != digest(root/"manifest.json")
                or manifest["source_file_sha256"] != staging["sequences"][name]["file_sha256"]
                or digest(path/"geometry.json") != manifest["geometry_sha256"]
                or digest(path/"evaluation.jsonl") != manifest["export_sha256"]):
            raise ValueError("per-video provenance changed")
        input_hashes[name] = digest(path/"manifest.json")
        for item in manifest["records"]:
            row = dict(item, partition="test", geometry_status=item["status"], status=item["scoring_status"])
            if row["status"] not in {"labeled", "borderline_unlabeled"}:
                row["status"] = "unavailable"
            all_records.append(row)
    result = dict(application="camera_projection_transfer", prompt=PROMPT,
                  scope="reviewed-point correspondence references for predicted camera-marker positions; not staticness or camera usability",
                  history="same twenty previously evaluated test videos; frozen judge, no new training or selection",
                  partitions={n:"test" for n in names},
                  source_manifest_sha256=digest(root/"manifest.json"), protocol_sha256=digest(plan_path),
                  per_video_corpus_sha256=input_hashes,
                  track_counts={n:staging["sequences"][n]["points_shape"][0] for n in names},
                  planned_presentations=len(all_records), records=all_records)
    validate_camera_transfer_roster(result)
    if len({r["id"] for r in all_records}) != len(all_records):
        raise ValueError("duplicate presentation IDs")
    output.mkdir(parents=True)
    (output/"images").mkdir()
    for name in names:
        video, _ = load_video(root, name)
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
        for row in (r for r in all_records if r["sequence"] == name):
            row["source_patch_ncc"] = None
            row["patch_baseline_prediction"] = "uncertain"
            if "image" not in row:
                continue
            source = (corpora/f"{name}-v1"/row["image"]).resolve()
            if not source.is_relative_to((corpora/f"{name}-v1").resolve()) or digest(source) != row["image_sha256"]:
                raise ValueError("panel hash or path changed")
            relative = f"images/{row['id']}.png"
            shutil.copyfile(source, output/relative)
            row["image"] = relative
            ncc = correlation(patch(gray[row["source_frame"]], row["source_xy"]),
                              patch(gray[row["frame"]], row["candidate_xy"]))
            row["source_patch_ncc"] = ncc
            if ncc is not None and ncc >= .8:
                row["patch_baseline_prediction"] = "close"
        del video, gray
    rows = [dict(id=r["id"], image=r["image"], prompt=PROMPT, target=r["label"])
            for r in all_records if r["status"] == "labeled"]
    export = output/"test.jsonl"
    export.write_text("".join(json.dumps(r)+"\n" for r in rows))
    result.update(exports_sha256=dict(test=digest(export)), script_sha256=digest(Path(__file__)),
                  baseline_helper_sha256=digest(Path(__file__).with_name("probe_tapvid_identity.py")),
                  baseline_policy="prior 17x17 source-patch NCC >=0.8 predicts close; otherwise uncertain; no motion/staticness inference or far label from rejection; unchanged threshold, chosen for this transfer comparison after first-video labels were available",
                  counts=dict(Counter(r["label"] or r["status"] for r in all_records)))
    (output/"manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "corpora", "plan", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(combine(args.root, args.corpora, args.plan, args.output)["counts"])
