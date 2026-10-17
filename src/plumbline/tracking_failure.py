"""Experimental functional-DNF accounting; no VLM verdict is assumed validated.

The attempt roster, not the available predictions, determines the denominator.
This module never removes raw metrics or changes the existing leaderboard.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def compare_functional_reviews(
    attempt_ids: list[str],
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare functional judgments on a fixed structurally-valid roster.

    References are supplied externally; this function does not establish their
    independence or correctness. Missing labels remain distinct from abstention.
    Rates over all resolved references must accompany conditional error rates:
    a judge that always abstains has zero detected errors, not perfect accuracy.
    Mechanical failures belong in summarize_failure_review, not this comparison.
    """
    roster = set()
    for key in attempt_ids:
        if not isinstance(key, str) or not key.strip() or key in roster:
            raise ValueError("attempt IDs must be distinct nonempty strings")
        roster.add(key)

    def index(rows):
        result = {}
        for row in rows:
            key = row["attempt_id"]
            if key not in roster or key in result:
                raise ValueError("unknown attempt or duplicate judgment; aggregate reviewers first")
            if row.get("verdict") not in {"usable", "unusable", "abstain"}:
                raise ValueError("invalid functional verdict")
            for field in ("evidence_id", "reason"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"judgment requires {field}")
            result[key] = row["verdict"]
        return result

    predicted, reference = index(predictions), index(references)
    labels = ("usable", "unusable", "abstain", "missing")
    matrix = {r: {p: 0 for p in labels} for r in labels}
    for key in attempt_ids:
        matrix[reference.get(key, "missing")][predicted.get(key, "missing")] += 1
    classes = {}
    for truth, error in (("usable", "unusable"), ("unusable", "usable")):
        row = matrix[truth]
        total = sum(row.values())
        decided = row["usable"] + row["unusable"]
        unresolved = row["abstain"] + row["missing"]
        classes[truth] = {
            "reference_n": total,
            "decided_n": decided,
            "error_n": row[error],
            "unresolved_n": unresolved,
            "coverage": decided / total if total else None,
            "error_rate_all_references": row[error] / total if total else None,
            "error_rate_decided": row[error] / decided if decided else None,
            "error_rate_bounds": [row[error] / total, (row[error] + unresolved) / total]
            if total else None,
        }
    return {
        "n_attempts": len(attempt_ids),
        "confusion_reference_rows_prediction_columns": matrix,
        "by_reference_class": classes,
        "interpretation": (
            "Usable-reference errors are false DNF; unusable-reference errors are missed failures. "
            "Bounds assign unresolved predictions to correct or incorrect outcomes; they are not "
            "confidence intervals. Unresolved references are not scored. Reference validity must "
            "be established separately."
        ),
    }


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
