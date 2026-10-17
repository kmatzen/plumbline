"""Validate independent reviewer exports and summarize canonical pair judgments.

Repeated presentations measure within-reviewer consistency and never add votes
to consensus. Abstentions are reported separately. An equal split is unresolved,
not an invented tie. This script does not synthesize or infer human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def summarize(study: dict, sessions: list[dict]) -> dict:
    presentations = {p["presentation_id"]: p for p in study["presentations"]}
    if len(presentations) != len(study["presentations"]):
        raise ValueError("duplicate presentation IDs")
    originals = {}
    for p in presentations.values():
        if p["system_a"] == p["system_b"]:
            raise ValueError("a comparison must contain distinct systems")
        if not p["repeat"]:
            if p["pair_key"] in originals:
                raise ValueError("multiple original presentations for one pair")
            originals[p["pair_key"]] = {p["system_a"], p["system_b"]}
    if len(originals) != study["unique_pairs"]:
        raise ValueError("unique pair count does not match study")
    for p in presentations.values():
        if originals.get(p["pair_key"]) != {p["system_a"], p["system_b"]}:
            raise ValueError("repeat has missing original or inconsistent systems")
    reviewers = set()
    individual = {}
    votes: dict[str, list[str]] = defaultdict(list)
    repeat_checks = []
    count = 0
    abstentions = 0
    for session in sessions:
        reviewer = session.get("reviewer", "").strip()
        if not reviewer or reviewer in reviewers:
            raise ValueError("one export per distinct nonempty reviewer code is required")
        reviewers.add(reviewer)
        individual[reviewer] = {}
        if session.get("study_id") != study["study_id"]:
            raise ValueError("response belongs to a different study")
        seen = set()
        judgments = defaultdict(list)
        for response in session["responses"]:
            pid = response["presentation_id"]
            if pid not in presentations or pid in seen:
                raise ValueError("unknown or duplicate presentation in response")
            seen.add(pid)
            presentation = presentations[pid]
            winner = response["winner"]
            if winner not in {"a", "b", "tie", "both_bad", "abstain"}:
                raise ValueError("invalid verdict")
            canonical = presentation["system_" + winner] if winner in {"a", "b"} else winner
            count += 1
            abstentions += winner == "abstain"
            judgments[presentation["pair_key"]].append(canonical)
            if not presentation["repeat"]:
                votes[presentation["pair_key"]].append(canonical)
                individual[reviewer][presentation["pair_key"]] = canonical
        for key, outcomes in judgments.items():
            if len(outcomes) > 1:
                repeat_checks.append(
                    {
                        "reviewer": reviewer,
                        "pair_key": key,
                        "agreement": len(set(outcomes)) == 1,
                        "contains_abstention": "abstain" in outcomes,
                    }
                )
    consensus = []
    for pair_key, outcomes in sorted(votes.items()):
        usable = [v for v in outcomes if v != "abstain"]
        counts = Counter(usable)
        leading, support = counts.most_common(1)[0] if counts else (None, 0)
        # A single vote is not independent corroboration. Require >=2 usable
        # reviewers and a strict majority of all usable labels, including ties.
        winner = leading if len(usable) >= 2 and support > len(usable) / 2 else None
        consensus.append(
            {
                "pair_key": pair_key,
                "verdict": winner,
                "n_reviewers": len(outcomes),
                "n_usable": len(usable),
                "counts": dict(counts),
            }
        )
    agreement = []
    for a, b in combinations(sorted(individual), 2):
        shared = individual[a].keys() & individual[b].keys()
        usable = [k for k in shared if "abstain" not in (individual[a][k], individual[b][k])]
        equal = sum(individual[a][k] == individual[b][k] for k in usable)
        agreement.append(
            {
                "reviewers": [a, b],
                "shared_original_pairs": len(shared),
                "jointly_usable_pairs": len(usable),
                "exact_agreements": equal,
                "exact_agreement_rate": equal / len(usable) if usable else None,
            }
        )
    return {
        "study_id": study["study_id"],
        "reviewers": len(reviewers),
        "responses": count,
        "abstentions": abstentions,
        "consensus": consensus,
        "repeat_checks": repeat_checks,
        "inter_reviewer_agreement": agreement,
        "resolved_pairs": sum(c["verdict"] is not None for c in consensus),
        "unreviewed_pairs": study["unique_pairs"] - len(consensus),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("responses", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new analysis file")
    result = summarize(
        json.loads(args.study.read_text()),
        [json.loads(path.read_text()) for path in args.responses],
    )
    result["provenance"] = {
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "response_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.responses
        ],
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"consensus", "repeat_checks"}}, indent=2
        )
    )
