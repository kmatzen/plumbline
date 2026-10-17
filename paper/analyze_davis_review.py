"""Validate a single project-author pilot export, without creating truth labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def analyze(study: dict, submission: dict) -> dict:
    if submission.get("schema") != "davis-usability-pilot-v1":
        raise ValueError("unknown submission schema")
    if submission.get("study_id") != study["study_id"]:
        raise ValueError("study mismatch")
    if not isinstance(submission.get("reviewer"), str) or not submission["reviewer"].strip():
        raise ValueError("reviewer code required")
    assets = {a["asset_id"] for a in study["assets"]}
    presentations = study["presentations"]
    mapping = {p["presentation_id"]: p["asset_id"] for p in presentations}
    if len(mapping) != len(presentations) or set(mapping.values()) != assets:
        raise ValueError("invalid study mapping")
    responses = submission.get("responses")
    if not isinstance(responses, list):
        raise ValueError("responses must be a list")
    seen = set()
    grouped = defaultdict(list)
    for response in responses:
        pid = response.get("presentation_id")
        if pid not in mapping or pid in seen:
            raise ValueError("unknown or duplicate presentation")
        seen.add(pid)
        if response.get("verdict") not in {"usable", "unusable", "abstain"}:
            raise ValueError("invalid verdict")
        for field in ("reason", "recorded_at"):
            if not isinstance(response.get(field), str) or not response[field].strip():
                raise ValueError(f"{field} required")
        if not isinstance(response.get("frame_indices_text"), str):
            raise ValueError("frame indices must remain free text")
        timestamp = datetime.fromisoformat(response["recorded_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timestamp needs timezone")
        grouped[mapping[pid]].append(response)
    planned = Counter(mapping.values())
    counts = Counter()
    repeated = []
    for asset in sorted(assets):
        answers = grouped[asset]
        verdicts = {r["verdict"] for r in answers}
        # No majority voting: an inconsistent repeat stays unresolved.
        status = (
            next(iter(verdicts))
            if len(verdicts) == 1
            else ("inconsistent" if verdicts else "unreviewed")
        )
        counts[status] += 1
        if planned[asset] > 1:
            repeated.append(
                {
                    "asset_id": asset,
                    "planned": planned[asset],
                    "answered": len(answers),
                    "all_answered": len(answers) == planned[asset],
                    "agreement": len(verdicts) == 1 if len(answers) > 1 else None,
                }
            )
    return {
        "study_id": study["study_id"],
        "scope": "single project-author development pilot; not independent validation",
        "presentations_planned": len(mapping),
        "presentations_answered": len(responses),
        "presentation_verdict_counts": dict(Counter(r["verdict"] for r in responses)),
        "unique_output_status_counts": dict(counts),
        "repeat_checks": repeated,
        "limitations": [
            "Usability judgments are not ground-truth trajectory errors or validated DNF labels.",
            "Repeated presentations are not independent observations.",
            "Partial reviews are retained; missing answers are not abstentions.",
            "No accuracy, confidence intervals, or inter-rater agreement are estimated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.study.read_text()), json.loads(args.submission.read_text()))
    result["input_sha256"] = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (("study", args.study), ("submission", args.submission))
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
