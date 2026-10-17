#!/usr/bin/env python3
"""Check that quantitative claims in the paper match the canonical site ledger."""

from __future__ import annotations

import hashlib
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

run = ROOT / "paper/results/davis/davis-dage-bear-v2"
manifest = json.loads((run / "manifest.json").read_text())
assert manifest["status"] == "complete"
assert len(manifest["frames"]) == 60
assert (
    hashlib.sha256((run / "prediction.npz").read_bytes()).hexdigest()
    == manifest["prediction_file_sha256"]
)
assert round(manifest["elapsed_seconds_including_load"], 2) == 72.08
evidence = json.loads((run / "image-evidence.json").read_text())
assert (
    evidence["source_manifest_sha256"]
    == hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
)
assert len(evidence["pairs"]) == 28
assert sum(p["count"] for p in evidence["pairs"]) == 22472
probe = json.loads((run / "rotation-probe.json").read_text())
assert (
    probe["evidence_sha256"]
    == hashlib.sha256((run / "image-evidence.json").read_bytes()).hexdigest()
)
assert len(probe["rows"]) == 588
for axis in range(3):
    baseline = {
        (r["a"], r["b"]): r["median"]
        for r in probe["rows"]
        if r["axis"] == axis and r["angle_degrees"] == 0
    }
    shifted = [r for r in probe["rows"] if r["axis"] == axis and r["angle_degrees"] == 15]
    assert len(shifted) == 28
    assert all(r["median"] > baseline[(r["a"], r["b"])] for r in shifted)
controls = ROOT / "paper/results/geometry-controls"
for name in ("on", "off"):
    result = json.loads((controls / f"control_{name}-qwen3b-768.json").read_text())
    assert result["status"] == "complete" and result["parse_error"] == "invalid_json"
    assert result["parsed_response"] is None
    # Inspect raw fenced answers for this audit only; never accept them as labels.
    raw = result["raw_response"].strip()
    assert raw.startswith("```json") and raw.endswith("```")
    assert json.loads(raw[7:-3])["verdict"] == "inconsistent"
failed = json.loads((controls / "control_on-qwen3b.json").read_text())
assert failed["status"] == "failed" and "OutOfMemoryError" in failed["traceback"]
for phrase in ("22,472", "28 frame pairs", "588", "72.08", "768-token"):
    assert phrase in TEX, f"missing checked development phrase: {phrase}"

print(
    "Selected manuscript counts match retained records; human validation and general claims are not certified"
)
