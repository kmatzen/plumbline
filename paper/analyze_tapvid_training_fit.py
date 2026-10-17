"""Analyze saved-adapter training fit; never treat it as held-out accuracy."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from paper.analyze_tapvid_lora import metrics, within_video_association
from paper.evaluate_tapvid_lora import parse_label


def analyze(report, training, corpus):
    if report["status"] != "complete" or report.get("partition") != "train":
        raise ValueError("complete training-fit report required")
    if training["status"] != "complete":
        raise ValueError("completed training required")
    selected = {r["id"]: r for r in corpus["records"]
                if r["partition"] == "train" and r["status"] == "labeled"}
    if set(training["training_ids"]) != set(selected) or len(training["training_ids"]) != len(selected):
        raise ValueError("training membership mismatch")
    output = report["results"]["adapted"]
    if len(output) != len(selected) or {r["id"] for r in output} != set(selected):
        raise ValueError("missing, duplicate, or extra training-fit predictions")
    rows = []
    for item in output:
        target = selected[item["id"]]
        if (item["target_for_scoring_only"] != target["label"]
                or item["sequence"] != target["sequence"]
                or parse_label(item["raw_response"]) != item["parsed"]):
            raise ValueError("target, sequence, or parser mismatch")
        rows.append(dict(id=item["id"], sequence=target["sequence"],
                         target=target["label"], answer=item["parsed"]))
    counts = Counter(r["target"] for r in rows)
    majority = sorted(counts, key=lambda c: (-counts[c], c))[0]
    return {"scope": "in-sample diagnostic; not validation or camera-DNF accuracy",
            "all": metrics(rows),
            "per_video": {name: metrics([r for r in rows if r["sequence"] == name])
                          for name in sorted({r["sequence"] for r in rows})},
            "training_majority_label": majority,
            "training_majority_accuracy": counts[majority]/len(rows),
            "within_video_association": within_video_association(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "training_manifest", "corpus_manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous analysis")
    report, training, corpus = [json.loads(p.read_text()) for p in
                                (args.report, args.training_manifest, args.corpus_manifest)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if report["training_manifest_sha256"] != digest(args.training_manifest):
        raise ValueError("training manifest changed")
    if training["corpus_manifest_sha256"] != digest(args.corpus_manifest):
        raise ValueError("corpus manifest changed")
    if report["evaluation_export_sha256"] != corpus["exports_sha256"]["train"]:
        raise ValueError("training export changed")
    result = analyze(report, training, corpus)
    result["source_sha256"] = {str(p): digest(p) for p in
                              (args.report, args.training_manifest, args.corpus_manifest, Path(__file__),
                               Path(__file__).with_name("analyze_tapvid_lora.py"))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
