"""Exhaustive paired track-subset sensitivity, not a sampling confidence interval."""

import argparse
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

from paper.analyze_layered_camera_ranking import compare, median_bracket
from paper.stage_static_rgbd_control import digest


def audit_pair(first, second):
    ids = [s["slot"] for s in first]
    if not ids or len(set(ids)) != len(ids) or ids != [s["slot"] for s in second]:
        raise ValueError("unique matching ordered track rosters required")
    if any(a["indices"] != b["indices"] for a, b in zip(first, second)):
        raise ValueError("paired observation indices differ")
    if len(ids) > 12:
        raise ValueError("exhaustive audit limited to twelve tracks")
    records, summaries = [], []
    labels = {"changed_higher_error": "first_lower", "changed_lower_error": "second_lower",
              "not_separated": "not_separated"}
    for size in range(len(ids), 0, -1):
        counts = Counter()
        for positions in itertools.combinations(range(len(ids)), size):
            a = median_bracket([first[i] for i in positions])
            b = median_bracket([second[i] for i in positions])
            verdict = labels[compare(a, b)]
            counts[verdict] += 1
            records.append(dict(slots=[ids[i] for i in positions], first=a, second=b, verdict=verdict))
        summaries.append(dict(retained_tracks=size, removed_tracks=len(ids)-size,
                              subsets=sum(counts.values()), counts=dict(counts)))
    full = records[0]["verdict"]
    contrary = [r for r in records if r["verdict"] != full]
    opposite = {"first_lower": "second_lower", "second_lower": "first_lower"}.get(full)
    reversed_rows = [r for r in records if r["verdict"] == opposite]
    return dict(full_roster_verdict=full, by_retained_count=summaries,
                minimum_removals_to_change_verdict=min((len(ids)-len(r["slots"]) for r in contrary), default=None),
                minimum_removals_to_reverse_preference=min((len(ids)-len(r["slots"]) for r in reversed_rows), default=None),
                subsets=records)


def run(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    if [c["model"] for c in data["cases"]] != ["dage", "da3"]:
        raise ValueError("expected complete ordered DAGE/DA3 pair")
    if any([s["slot"] for s in c["slots"]] != list(range(12)) for c in data["cases"]):
        raise ValueError("expected all twelve original slots")
    return dict(scope="development sensitivity of a classical geometric ranking to observed track selection",
                policy="enumerate every nonempty paired subset, including the complete roster; identical slots and observations for both methods; unresolved slots remain [0,infinity]; no subset selected as replacement score",
                limitations="not independent samples or a confidence interval; does not cover new unobserved tracks; no camera accuracy or DNF labels; exploratory analysis after full-roster results",
                first_model="dage", second_model="da3", source_sha256=digest(source),
                script_sha256=digest(Path(__file__)),
                aggregate_sha256=digest(Path(__file__).with_name("analyze_layered_camera_ranking.py")),
                **audit_pair(*(c["slots"] for c in data["cases"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.source)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write((json.dumps(result, allow_nan=False)+"\n").encode())
    print(json.dumps({k: v for k, v in result.items() if k != "subsets"}, indent=2))
