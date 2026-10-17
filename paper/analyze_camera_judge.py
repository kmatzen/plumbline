"""Audit unlabeled camera-marker responses without treating them as truth."""

import argparse
import json
from collections import Counter
from pathlib import Path

from paper.evaluate_tapvid_lora import parse_label
from paper.stage_static_rgbd_control import digest


def summarize(report, corpus):
    if report["status"] != "complete" or report["partition"] != "camera":
        raise ValueError("complete camera application required")
    records = corpus["records"]
    expected = {(m, s, f) for m in ("dage", "da3") for s in range(12) for f in (30, 60, 89)}
    if (len(records) != 72 or len({r["id"] for r in records}) != 72
            or {(r["model"], r["slot"], r["frame"]) for r in records} != expected):
        raise ValueError("full unique presentation roster required")
    selected = {r["id"]:r for r in records if r["status"] == "renderable"}
    if (report["planned_examples"] != 72 or report["renderable_examples"] != len(selected)
            or report["unavailable_counts"] != dict(Counter(r["status"] for r in records))):
        raise ValueError("availability accounting changed")
    indexed, arms = {}, {}
    for arm in ("base", "adapted"):
        rows = report["results"][arm]
        if len(rows) != len(selected) or {r["id"] for r in rows} != set(selected):
            raise ValueError("missing, duplicate, or extra responses")
        indexed[arm] = {r["id"]:r for r in rows}
        for row in rows:
            if row["target_for_scoring_only"] is not None or row["parsed"] != parse_label(row["raw_response"]):
                raise ValueError("unexpected target or parser mismatch")
            if not isinstance(row["image_tokens"], int) or row["image_tokens"] <= 0:
                raise ValueError("invalid image token count")
        summaries = {}
        for model in ("dage", "da3"):
            planned = [r for r in records if r["model"] == model]
            observed = [indexed[arm][r["id"]] for r in planned if r["status"] == "renderable"]
            summaries[model] = dict(planned_presentations=len(planned), presented=len(observed),
                                   unavailable=dict(Counter(r["status"] for r in planned if r["status"] != "renderable")),
                                   predicted_labels=dict(Counter(r["parsed"] for r in observed)),
                                   interpretation="prediction counts, not accuracy or camera failure counts")
        arms[arm] = summaries
    for key in selected:
        a, b = indexed["base"][key], indexed["adapted"][key]
        if (a["processed_pixel_sha256"] != b["processed_pixel_sha256"]
                or a["image_tokens"] != b["image_tokens"]):
            raise ValueError("paired image inputs differ")
    pairs = [dict(id=r["id"], model=r["model"], slot=r["slot"], frame=r["frame"],
                  base=indexed["base"][r["id"]]["parsed"], adapted=indexed["adapted"][r["id"]]["parsed"])
             for r in records if r["status"] == "renderable"]
    return dict(scope="unlabeled camera-marker correspondence predictions; no accuracy or DNF validation",
                planned_presentations=72, renderable_presentations=len(selected), arms=arms,
                base_to_adapted_counts=dict(Counter(f'{r["base"]}->{r["adapted"]}' for r in pairs)),
                paired_predictions=pairs,
                limitations="do not interpret close counts as camera ranking or far as DNF; unverified prefix fits, calibration, correspondence, and staticness; missing cases remain unassessed; no independent labels")


def run(report_path, corpus_path, training_path, evaluator_path):
    report = json.loads(report_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    for key, path in (("evaluation_corpus_manifest_sha256", corpus_path),
                      ("training_manifest_sha256", training_path), ("script_sha256", evaluator_path)):
        if report[key] != digest(path):
            raise ValueError(f"changed provenance: {key}")
    if report["evaluation_export_sha256"] != corpus["export_sha256"]:
        raise ValueError("export mismatch")
    result = summarize(report, corpus)
    result["input_sha256"] = {name:digest(path) for name, path in
        (("report", report_path), ("corpus", corpus_path), ("training", training_path),
         ("evaluator", evaluator_path), ("analysis", Path(__file__)))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "corpus", "training", "evaluator", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.report, args.corpus, args.training, args.evaluator)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["arms"], indent=2))
