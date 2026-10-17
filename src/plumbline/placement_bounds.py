"""Missing-observation bounds for conditional placement consistency.

These are identification bounds, not confidence intervals or camera-DNF labels.
Measured errors are treated as valid observations; their correctness is a
separate assumption that this calculation cannot establish.
"""

import numpy as np


def paired_bounds(left, right, tolerance_pixels):
    """Bound the left-minus-right success fraction over all planned observations.

    Use None for unavailable evidence. No independence or missing-at-random
    assumption is made. Every unavailable binary outcome can range from zero to one.
    """
    if not np.isfinite(tolerance_pixels) or tolerance_pixels < 0:
        raise ValueError("finite nonnegative tolerance required")
    if len(left) != len(right) or not len(left):
        raise ValueError("nonempty matched planned observations required")

    def summarize(values):
        observed = [v for v in values if v is not None]
        if any(not np.isfinite(v) or v < 0 for v in observed):
            raise ValueError("errors must be finite nonnegative or None")
        successes = sum(v <= tolerance_pixels for v in observed)
        unknown = len(values) - len(observed)
        return {
            "measured": len(observed),
            "unknown": unknown,
            "observed_successes": successes,
            "success_fraction_bounds": [
                successes / len(values),
                (successes + unknown) / len(values),
            ],
        }

    a, b = summarize(left), summarize(right)
    lower = a["success_fraction_bounds"][0] - b["success_fraction_bounds"][1]
    upper = a["success_fraction_bounds"][1] - b["success_fraction_bounds"][0]
    return {
        "planned": len(left),
        "tolerance_pixels": float(tolerance_pixels),
        "left": a,
        "right": b,
        "left_minus_right_bounds": [lower, upper],
        "ordering": "left"
        if lower > 0
        else "right"
        if upper < 0
        else "equal"
        if lower == upper == 0
        else "unresolved",
        "assumption": "observed image tracks and placement errors are valid; unmeasured outcomes unconstrained",
    }
