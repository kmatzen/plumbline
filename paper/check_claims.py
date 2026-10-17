#!/usr/bin/env python3
"""Check that quantitative claims in the paper match the canonical site ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "site" / "explore.html").read_text()
TEX = (ROOT / "paper" / "main.tex").read_text()
SENSITIVITY = json.loads((ROOT / "paper" / "results" / "sensitivity.json").read_text())
SEARCH = json.loads((ROOT / "paper" / "results" / "design_search.json").read_text())


def load_ledger(name: str):
    match = re.search(
        rf'<script type="application/json" id="{re.escape(name)}">\n([\s\S]*?)\n</script>',
        HTML,
    )
    if not match:
        raise RuntimeError(f"could not find {name}")
    return json.loads(match.group(1))


cells = load_ledger("verified-cells")
assert len(cells) == 47
assert sum(c["task"] == "depth" for c in cells) == 39
assert sum(c["task"] == "pose" for c in cells) == 8
assert len({c["model"] for c in cells}) == 16
assert len({c["ds"] for c in cells}) == 13

deviations = sorted(abs(float(c["d"])) for c in cells)
median = deviations[len(deviations) // 2]
p90 = deviations[int(0.9 * (len(deviations) - 1))]
within_five = sum(value <= 5.0 + 1e-9 for value in deviations)

assert round(median, 1) == 1.5
assert round(p90, 1) == 4.8
assert within_five == 46
assert round(max(deviations), 1) == 7.3

for phrase in ("47 verified", "39 depth", "8 pose", "16 method", "13 dataset", "46 of 47"):
    assert phrase in TEX, f"missing checked manuscript phrase: {phrase}"

assert len(SENSITIVITY["rows"]) == 480
assert SENSITIVITY["summary"]["rotation"]["ate"]["tie_fraction"] == 1.0
assert SENSITIVITY["summary"]["rotation"]["rpe_rotation"]["increasing_amplitude_concordance"] == 1.0
assert "92.9" not in TEX and "79.6" not in TEX, "withdrawn comparison restored"
assert SEARCH["trials"] == 60
assert round(SEARCH["bayes_success_at_20"] * 100, 1) == 48.3
assert round(SEARCH["random_success_at_20"] * 100, 1) == 10.0
for phrase in ("480", "48.3", "10.0", "60 trials"):
    assert phrase in TEX, f"missing checked experiment phrase: {phrase}"

print("paper claims match the canonical ledger")
