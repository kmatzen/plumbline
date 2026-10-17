"""Predeclared paired video-level analysis for the fixed twenty-pass judge."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.analyze_tapvid_lora import compare
from paper.evaluate_tapvid_lora import digest


def paired_video_interval(base, adapted):
    if set(base) != set(adapted) or not base:
        raise ValueError("matching nonempty video sets required")
    names = sorted(base)
    differences = np.array([adapted[n]-base[n] for n in names], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("finite per-video accuracies required")
    rng = np.random.default_rng(677)
    samples = differences[rng.integers(0, len(names), size=(20000, len(names)))].mean(axis=1)
    return {"video_count": len(names), "macro_accuracy_difference": float(differences.mean()),
            "percentile_interval_95": np.quantile(samples, [.025,.975]).tolist(),
            "resamples": 20000, "seed": 677,
            "interpretation": "paired resampling of whole videos; equal weight per video with labeled examples; descriptive uncertainty, not independent-frame significance or proof of generalization"}


def analyze(report, corpus, development, tracker):
    result = compare(report, corpus, tracker, partition="test", training_corpus=development)
    planned = len(corpus["records"])
    statuses = Counter(r["status"] for r in corpus["records"])
    names = set(corpus["partitions"])
    result["roster"] = {"planned_examples": planned, "statuses": dict(statuses),
                        "planned_videos": len(names),
                        "interpretation": "borderline and missing examples have no scored response; they are neither correct nor incorrect"}
    for arm in result["arms"].values():
        arm["answered_fraction_of_planned"] = arm["all"]["answered_examples"]/planned if planned else None
        arm["close_fraction_of_planned"] = arm["all"]["close_predictions"]/planned if planned else None
        arm["videos_without_labeled_examples"] = sorted(names-set(arm["per_video"]))
    result["paired_video_uncertainty"] = paired_video_interval(
        {n:v["accuracy_all_examples"] for n,v in result["arms"]["base"]["per_video"].items()},
        {n:v["accuracy_all_examples"] for n,v in result["arms"]["adapted"]["per_video"].items()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "training", "corpus", "development", "tracker", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior analysis")
    report, training, corpus, development = [json.loads(p.read_text()) for p in
                                           (args.report,args.training,args.corpus,args.development)]
    if training["status"] != "complete" or training.get("epochs") != 20:
        raise ValueError("completed fixed twenty-pass adapter required")
    for actual, path in ((report["training_manifest_sha256"],args.training),
                         (report["evaluation_corpus_manifest_sha256"],args.corpus),
                         (report["development_manifest_sha256"],args.development),
                         (training["corpus_manifest_sha256"],args.development),
                         (corpus["predictions_sha256"],args.tracker)):
        if actual != digest(path):
            raise ValueError("analysis input hash mismatch")
    tracker = json.loads(gzip.decompress(args.tracker.read_bytes()))
    result = analyze(report,corpus,development,tracker)
    result["source_sha256"] = {str(p):digest(p) for p in
        (args.report,args.training,args.corpus,args.development,args.tracker,Path(__file__),
         Path(__file__).with_name("analyze_tapvid_lora.py"),Path(__file__).with_name("evaluate_tapvid_lora.py"))}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
