"""Independent correspondence accuracy for the fixed camera-position transfer."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from paper.analyze_tapvid_external import paired_video_interval
from paper.analyze_tapvid_lora import metrics
from paper.evaluate_tapvid_lora import digest, parse_label, validate_camera_transfer_roster, validate_test_partition


def analyze(report, corpus, development):
    if (report["status"] != "complete" or report["partition"] != "test"
            or report.get("application") != "camera_projection_transfer"
            or corpus.get("application") != "camera_projection_transfer"):
        raise ValueError("complete camera-position transfer comparison required")
    validate_test_partition(corpus, development)
    validate_camera_transfer_roster(corpus)
    selected = {r["id"]:r for r in corpus["records"] if r["status"] == "labeled"}
    if report["planned_examples"] != len(corpus["records"]) or report["labeled_examples"] != len(selected):
        raise ValueError("planned/labeled accounting mismatch")
    arms, paired = {}, {}
    for arm in ("base", "adapted"):
        predictions = report["results"][arm]
        if len(predictions) != len(selected) or {p["id"] for p in predictions} != set(selected):
            raise ValueError("missing, duplicate, or extra predictions")
        rows = []
        for item in predictions:
            reference = selected[item["id"]]
            if (item["target_for_scoring_only"] != reference["label"]
                    or item["sequence"] != reference["sequence"]
                    or parse_label(item["raw_response"]) != item["parsed"]
                    or type(item["image_tokens"]) is not int or item["image_tokens"] <= 0):
                raise ValueError("reference, parser, or image-token mismatch")
            image = (item["processed_pixel_sha256"], item["image_tokens"])
            if arm == "base":
                paired[item["id"]] = image
            elif image != paired[item["id"]]:
                raise ValueError("paired processed images differ")
            rows.append(dict(id=item["id"], sequence=reference["sequence"], model=reference["model"],
                             target=reference["label"], answer=item["parsed"]))
        arms[arm] = rows
    training_counts = Counter(r["label"] for r in development["records"]
                              if r["partition"] == "train" and r["status"] == "labeled")
    if not training_counts:
        raise ValueError("training-only majority baseline unavailable")
    majority = sorted(training_counts, key=lambda label:(-training_counts[label], label))[0]
    arms["training_majority"], arms["source_patch_ncc"] = [], []
    for row in selected.values():
        common = dict(id=row["id"], sequence=row["sequence"], model=row["model"], target=row["label"])
        ncc = row["source_patch_ncc"]
        if ncc is not None and not (math.isfinite(ncc) and -1 <= ncc <= 1):
            raise ValueError("invalid patch correlation")
        answer = "close" if ncc is not None and ncc >= .8 else "uncertain"
        if answer != row["patch_baseline_prediction"]:
            raise ValueError("patch baseline policy changed")
        arms["source_patch_ncc"].append(dict(common, answer=answer))
        arms["training_majority"].append(dict(common, answer=majority))
    names = sorted(corpus["partitions"])
    planned = len(corpus["records"])
    summaries = {}
    for arm, rows in arms.items():
        per_video = {n:metrics([r for r in rows if r["sequence"] == n]) for n in names}
        observed = [v["accuracy_all_examples"] for v in per_video.values() if v["examples"]]
        summary = metrics(rows)
        summaries[arm] = dict(all=summary, per_video=per_video,
            by_candidate_method={m:metrics([r for r in rows if r["model"] == m]) for m in ("dage", "da3")},
            macro_accuracy_on_videos_with_labels=sum(observed)/len(observed) if observed else None,
            videos_without_labeled_examples=[n for n,v in per_video.items() if not v["examples"]],
            answered_fraction_of_planned=summary["answered_examples"]/planned if planned else None)
    base = {n:v["accuracy_all_examples"] for n,v in summaries["base"]["per_video"].items() if v["examples"]}
    adapted = {n:summaries["adapted"]["per_video"][n]["accuracy_all_examples"] for n in base}
    return dict(scope="independently labeled projected-position correspondence accuracy; not camera quality or DNF",
                roster=dict(planned_presentations=planned, labeled_presentations=len(selected),
                            label_coverage=len(selected)/planned if planned else None,
                            statuses=dict(Counter(r["status"] for r in corpus["records"])),
                            reference_labels=dict(Counter(r["label"] for r in selected.values()))),
                arms=summaries, training_majority_label=majority,
                paired_video_uncertainty=paired_video_interval(base, adapted) if len(base) >= 2 else None,
                limitations="previously evaluated videos; dependent frames/tracks/methods grouped by video; accuracy excludes unavailable and borderline references, whose counts remain reported; absent reference classes make balanced accuracy unavailable; no camera-level DNF labels; NCC baseline added after first-video label inspection")


def run(report_path, training_path, corpus_path, development_path, evaluator_path):
    report, training, corpus, development = [json.loads(p.read_text()) for p in
                                            (report_path, training_path, corpus_path, development_path)]
    if training["status"] != "complete" or training.get("epochs") != 20:
        raise ValueError("fixed complete twenty-pass training required")
    checks = ((report["training_manifest_sha256"], training_path),
              (report["evaluation_corpus_manifest_sha256"], corpus_path),
              (report["development_manifest_sha256"], development_path),
              (report["script_sha256"], evaluator_path),
              (training["corpus_manifest_sha256"], development_path))
    if any(actual != digest(path) for actual,path in checks):
        raise ValueError("analysis provenance mismatch")
    if report["evaluation_export_sha256"] != corpus["exports_sha256"]["test"]:
        raise ValueError("evaluation export mismatch")
    result = analyze(report, corpus, development)
    result["source_sha256"] = {name:digest(path) for name,path in
        (("report",report_path),("training",training_path),("corpus",corpus_path),
         ("development",development_path),("evaluator",evaluator_path),("analysis",Path(__file__)),
         ("metrics",Path(__file__).with_name("analyze_tapvid_lora.py")),
         ("video_interval",Path(__file__).with_name("analyze_tapvid_external.py")))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "training", "corpus", "development", "evaluator", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.report, args.training, args.corpus, args.development, args.evaluator)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["roster"], indent=2))
