"""Complete-case validation reporting with abstentions in the denominator."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from paper.evaluate_tapvid_lora import parse_label

TARGETS = ("close", "far", "occluded")
ANSWERS = TARGETS + ("uncertain", "invalid")


def metrics(rows):
    matrix = {target: {answer: 0 for answer in ANSWERS} for target in TARGETS}
    for row in rows:
        matrix[row["target"]][row["answer"]] += 1
    n = len(rows)
    correct = sum(matrix[c][c] for c in TARGETS)
    recall = {c: matrix[c][c]/sum(matrix[c].values()) if sum(matrix[c].values()) else None for c in TARGETS}
    close_count = sum(matrix[c]["close"] for c in TARGETS)
    answered = sum(matrix[c][a] for c in TARGETS for a in TARGETS)
    return {"examples": n, "confusion": matrix,
            "accuracy_all_examples": correct/n if n else None,
            "class_recall": recall,
            "balanced_accuracy": sum(recall.values())/3 if all(v is not None for v in recall.values()) else None,
            "answered_examples": answered,
            "answer_coverage": answered/n if n else None,
            "accuracy_answered_only": correct/answered if answered else None,
            "uncertain_examples": sum(matrix[c]["uncertain"] for c in TARGETS),
            "invalid_examples": sum(matrix[c]["invalid"] for c in TARGETS),
            "close_predictions": close_count,
            "close_precision": matrix["close"]["close"]/close_count if close_count else None,
            "false_close_predictions": matrix["far"]["close"]+matrix["occluded"]["close"]}


def within_video_association(rows):
    """Exact expected accuracy after uniformly permuting answers within videos.

    This preserves each video's answer frequencies, including abstentions. It
    is a descriptive control for video-level label priors, not a permutation
    significance test: observations within a video/track are dependent.
    """
    videos = sorted({r["sequence"] for r in rows})
    details = {}
    for video in videos:
        subset = [r for r in rows if r["sequence"] == video]
        targets = Counter(r["target"] for r in subset)
        answers = Counter(r["answer"] for r in subset)
        n = len(subset)
        expected = sum(targets[c]*answers[c] for c in TARGETS)/(n*n)
        observed = sum(r["target"] == r["answer"] for r in subset)/n
        details[video] = dict(examples=n, observed_accuracy=observed,
                             shuffled_expected_accuracy=expected,
                             excess_accuracy=observed-expected)
    return {"per_video": details,
            "macro_video_excess_accuracy": sum(d["excess_accuracy"] for d in details.values())/len(details) if details else None,
            "interpretation": "within-video answer/target association beyond marginal label frequencies; not causal image reliance, a trained baseline, or a significance test"}


def compare(report, corpus, tracker):
    if report.get("partition", "validation") != "validation":
        raise ValueError("training-fit results are not validation evidence")
    if report["status"] != "complete":
        raise ValueError("incomplete generation run")
    selected = {r["id"]: r for r in corpus["records"] if r["partition"] == "validation" and r["status"] == "labeled"}
    arms = {}
    pixels = {}
    for arm in ("base", "adapted"):
        output = report["results"][arm]
        if len(output) != len(selected) or {r["id"] for r in output} != set(selected):
            raise ValueError("missing, duplicate, or extra validation predictions")
        rows = []
        for item in output:
            target = selected[item["id"]]
            if (item["target_for_scoring_only"] != target["label"] or item["sequence"] != target["sequence"]
                    or parse_label(item["raw_response"]) != item["parsed"]):
                raise ValueError("target or parser mismatch")
            key = item["id"]
            if arm == "base":
                pixels[key] = item["processed_pixel_sha256"]
            elif item["processed_pixel_sha256"] != pixels[key]:
                raise ValueError("paired images differ")
            rows.append({"id": key, "sequence": target["sequence"], "target": target["label"], "answer": item["parsed"]})
        arms[arm] = rows
    counts = Counter(r["label"] for r in corpus["records"] if r["partition"] == "train" and r["status"] == "labeled")
    majority = sorted(counts, key=lambda c: (-counts[c], c))[0]
    baseline, close_gate = [], []
    for key, item in selected.items():
        common = dict(id=key, sequence=item["sequence"], target=item["label"])
        baseline.append(dict(**common, answer=majority))
        track = tracker["sequences"][item["sequence"]]["tracks"][item["track"]]
        if track["track"] != item["track"]:
            raise ValueError("tracker slot changed")
        row = next(r for r in track["records"] if r["frame"] == item["frame"])
        if row["xy"] != item["candidate_xy"]:
            raise ValueError("candidate changed")
        accept = (row["xy"] is not None and row["forward_backward_pixels"] is not None
                  and row["forward_backward_pixels"] <= 1 and row["source_patch_ncc"] is not None
                  and row["source_patch_ncc"] >= .8)
        close_gate.append(dict(**common, answer="close" if accept else "uncertain"))
    arms["training_majority"] = baseline
    arms["frozen_image_close_gate"] = close_gate
    results = {}
    for arm, rows in arms.items():
        per_video = {name: metrics([r for r in rows if r["sequence"] == name]) for name in sorted({r["sequence"] for r in rows})}
        results[arm] = {"all": metrics(rows), "per_video": per_video,
                        "macro_video_accuracy": sum(v["accuracy_all_examples"] for v in per_video.values())/len(per_video),
                        "within_video_association": within_video_association(rows)}
    a = {r["id"]: r["answer"] == r["target"] for r in arms["base"]}
    b = {r["id"]: r["answer"] == r["target"] for r in arms["adapted"]}
    return {"scope": "two-video development validation; not independent frame-level evidence or camera DNF",
            "selection": "only preselected labeled candidate panels; excludes borderline and unavailable cases by corpus construction",
            "baseline_policy": "majority from training labels only; image check may predict close or abstain, never infer far from rejection",
            "training_majority_label": majority, "arms": results,
            "paired_correctness_counts": dict(Counter(f"base_{a[k]}_adapted_{b[k]}" for k in selected)),
            "limitations": "no scene-level significance claim from two videos; answered-only accuracy has a model-selected denominator and must be read with coverage and all-example accuracy; generation-format improvement is separate from class recognition; no calibration examples used"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "corpus_manifest", "tracker", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous analysis")
    report = json.loads(args.report.read_text())
    corpus = json.loads(args.corpus_manifest.read_text())
    tracker = json.loads(gzip.decompress(args.tracker.read_bytes()))
    if hashlib.sha256(args.tracker.read_bytes()).hexdigest() != corpus["predictions_sha256"]:
        raise ValueError("tracker input hash mismatch")
    result = compare(report, corpus, tracker)
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.report, args.corpus_manifest, args.tracker, Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
