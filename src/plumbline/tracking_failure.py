"""Experimental functional-DNF accounting; no VLM verdict is assumed validated.

The attempt roster, not the available predictions, determines the denominator.
This module never removes raw metrics or changes the existing leaderboard.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def summarize_failure_review(
    attempts: list[dict[str, Any]], judgments: list[dict[str, Any]], *, metric: str
) -> dict[str, Any]:
    """Summarize one system/configuration on a fixed roster of attempted tasks.

    Attempt status is complete, crash, timeout, invalid, or pending. Complete
    means structurally valid, not useful. A judgment is usable, unusable, or
    abstain and must cite an evidence ID and a reason. Missing judgments are
    unresolved, never implicit passes. Metric means are descriptive only.
    """
    roster = {}
    for row in attempts:
        key = row["attempt_id"]
        if not isinstance(key, str) or not key.strip() or key in roster:
            raise ValueError("attempt IDs must be distinct nonempty strings")
        if row.get("status") not in {"complete", "crash", "timeout", "invalid", "pending"}:
            raise ValueError("invalid attempt status")
        roster[key] = row
    decisions = {}
    for row in judgments:
        key = row["attempt_id"]
        if key not in roster or key in decisions:
            raise ValueError("unknown attempt or duplicate judgment; aggregate reviewers first")
        if roster[key]["status"] != "complete":
            raise ValueError("only complete valid attempts receive functional judgments")
        if row.get("verdict") not in {"usable", "unusable", "abstain"}:
            raise ValueError("invalid functional verdict")
        for field in ("evidence_id", "reason"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"judgment requires {field}")
        decisions[key] = row
    counts: Counter[str] = Counter()
    all_values, accepted_values = [], []
    for key, row in roster.items():
        status = row["status"]
        if status != "complete":
            counts[status] += 1
            continue
        verdict = decisions.get(key, {}).get("verdict", "unreviewed")
        counts[verdict] += 1
        value = row.get("metrics", {}).get(metric)
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("metric must be a finite number or missing")
            all_values.append(float(value))
            if verdict == "usable":
                accepted_values.append(float(value))
    total = len(roster)
    mechanical = sum(counts[s] for s in ("crash", "timeout", "invalid"))
    functional = counts["unusable"]
    unresolved = sum(counts[s] for s in ("pending", "abstain", "unreviewed"))
    return {
        "n_attempts": total,
        "counts": {
            s: counts[s]
            for s in (
                "usable",
                "unusable",
                "abstain",
                "unreviewed",
                "crash",
                "timeout",
                "invalid",
                "pending",
            )
        },
        "mechanical_dnf_rate": mechanical / total if total else None,
        "functional_dnf_rate": functional / total if total else None,
        "unresolved_rate": unresolved / total if total else None,
        "failure_rate_bounds": [
            (mechanical + functional) / total,
            (mechanical + functional + unresolved) / total,
        ]
        if total
        else None,
        "metric": metric,
        "all_complete_metric_n": len(all_values),
        "all_complete_metric_mean": sum(all_values) / len(all_values) if all_values else None,
        "accepted_metric_n": len(accepted_values),
        "accepted_metric_mean": sum(accepted_values) / len(accepted_values)
        if accepted_values
        else None,
        "interpretation": "Conditional means must be reported with failure and unresolved rates; not a ranking score.",
    }
