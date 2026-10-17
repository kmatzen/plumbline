#!/usr/bin/env python3
"""Check that quantitative claims in the paper match the canonical site ledger."""

from __future__ import annotations

import hashlib
import gzip
import json
import re
from pathlib import Path

import numpy as np

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
for phrase in ("22,472", "28 frame pairs", "588", "72.08"):
    assert phrase in TEX, f"missing checked development phrase: {phrase}"

for clip in ("bear", "bmx-bumps", "boat"):
    baseline_path = ROOT / f"paper/results/davis/grid-anchors-v1/{clip}.json"
    baseline = json.loads(baseline_path.read_text())
    co_dir = ROOT / f"paper/results/davis/cotracker-{clip}-v1"
    co_manifest = json.loads((co_dir / "manifest.json").read_text())
    co = json.loads((co_dir / "anchor-analysis.json").read_text())
    assert co_manifest["status"] == "complete"
    assert co["baseline_sha256"] == hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert co_manifest["baseline_sha256"] == co["baseline_sha256"]
    arrays = {}
    for phase in co_manifest["phases"]:
        path = co_dir / f"{phase['name']}.npz"
        assert phase["status"] == "complete"
        assert phase["prediction_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            arrays[phase["name"]] = data["tracks"]
    prefix_delta = np.linalg.norm(arrays["prefix"] - arrays["full"][:11], axis=-1)
    assert np.allclose(
        np.percentile(prefix_delta, [0, 50, 100]),
        co["prefix_full_coordinate_delta_quantiles_0_50_100"],
    )
    assert f"{prefix_delta.max():.3f}" in TEX
    for tracker, data in (("LK", baseline), ("CoTracker3", co)):
        errors = [[], []]
        anchor_count = 0
        for slot, track in enumerate(data["tracks"]):
            count = 0
            for frame in range(11, len(track)):
                values = [
                    method["anchors"][slot]["rows"][frame]["distance_pixels"]
                    for method in data["methods"]
                ]
                if track[frame].get("annotation_label") == 0 and all(v is not None for v in values):
                    count += 1
                    for index, value in enumerate(values):
                        errors[index].append(value)
            anchor_count += count > 0
        assert data["possible_test_observations"] == 588
        table_row = (
            f"{clip} & {tracker} & {len(errors[0])} & {anchor_count} & "
            f"{np.median(errors[0]):.3f} & {np.median(errors[1]):.3f}"
        )
        assert table_row in TEX, f"unmatched placement row: {table_row}"
    assert len(baseline["tracks"]) == 12
    if clip == "bmx-bumps":
        assert all(track[-1]["status"] == "lost" for track in baseline["tracks"])
    if clip == "boat":
        assert [m["fit_counts"]["fit"] for m in co["methods"]] == [10, 6]
        assert [m["fit_counts"]["nonpositive_fit_depth"] for m in co["methods"]] == [2, 6]

tum = json.loads((ROOT / "paper/results/dense/tum-anchor-xyz-v1.json").read_text())
assert tum["possible_later_observations"] == 948
common = tum["common_observations"]
assert len(common) == 130
later = [r for r in common if r["frame"] >= 31]
assert len(later) == 70
prefix = json.loads((ROOT / "paper/results/dense/tum-anchor-prefix-v1.json").read_text())
intersection = [r for r in prefix["rows"] if r["common"]]
assert len(intersection) == 49 and len(prefix["rows"]) == 708
ore_root = ROOT / "paper/results/ore-official-demo-v1"
depth = json.loads((ore_root / "depth-support-audit.json").read_text())
tracks = [t for m in depth["methods"] for t in m["tracks"]]
assert len(tracks) == 28
assert sum(t["official_nonpositive_frames"] > 0 for t in tracks) == 12
assert sum(t["positive_depth_candidate_count"] == 0 for t in tracks) == 9
assert sum(not t["positive_depth_interval"]["feasible"] for t in tracks) == 9
boxes = json.loads((ore_root / "box-preprocessing-v2.json").read_text())["tracks"]
assert [t["reversed_after_integer_cast"] for t in boxes] == [24, 371, 40, 429]
assert sum(t["frames"] for t in boxes) * 2 == 6774
assert sum(v > 1 for t in boxes for r in t["rows"] for v in r["roundtrip_error_pixels"]) == 3995
assert all(not r["reversed_before_distortion"] for t in boxes for r in t["rows"])

timing = json.loads((ROOT / "paper/results/dense/tum-pose-timing-xyz-v2.json").read_text())
assert timing["comparison_start_frame"] == 1
assert timing["fit_original_frames"] == list(range(1, 12))
assert len(timing["common_observations"]) == 141
paired = [
    abs(r["pixels"]["associated_reference"] - r["pixels"]["interpolated_reference"])
    for r in timing["common_observations"]
]
description_root = ROOT / "paper/results/description-controls-v1"
description = json.loads((description_root / "shapes.json").read_text())
identity = json.loads((description_root / "identity-repeat.json").read_text())
for key in (
    "image_sha256",
    "processed_input_audit",
    "checkpoint_files",
    "runner_sha256",
    "settings",
    "attention_helper_sha256",
    "attention_implementation_sha256",
):
    assert description[key] == identity[key]
assert description["processed_input_audit"]["actual_image_tokens"] == 680
assert description["processed_input_audit"]["expected_image_tokens"] == 680
assert description["processed_input_audit"]["pixel_finite"]
assert description["processed_input_audit"]["pixel_std"] > 0
assert identity["parsed_response"] is None and identity["parse_error"] == "invalid_json"
assert '"verdict": "abstain"' in identity["raw_response"]
arithmetic = json.loads(
    (ROOT / "paper/results/qwen7b-nf4-controls-v1/attention-audit-trace.json").read_text()
)["statistics"]
assert arithmetic["query"]["nonfinite"] == arithmetic["key"]["nonfinite"] == 0
assert arithmetic["qk_fp16"]["nonfinite"] > 0 and arithmetic["qk_fp32"]["nonfinite"] == 0

# Development failures remain audited above, but are not manuscript sections.
for excluded in (
    "Unsuccessful author-review",
    "failed VLM",
    "sec:vlmcontrol",
    "exclamation marks",
    "GPU out-of-memory",
    "Audit of the released ORE sample",
):
    assert excluded not in TEX, f"development-only narrative restored: {excluded}"

tapvid_root = ROOT / "paper/results/tapvid-davis-v1"
tapvid_raw = (tapvid_root / "identity-held-out-v1.json.gz").read_bytes()
tapvid = json.loads(gzip.decompress(tapvid_raw))
tapvid_summary = json.loads((tapvid_root / "identity-held-out-summary-v1.json").read_text())
assert tapvid_summary["source_sha256"] == hashlib.sha256(tapvid_raw).hexdigest()
assert len(tapvid["sequences"]) == 20
assert sum(len(s["tracks"]) for s in tapvid["sequences"].values()) == 435
for gate, label in (("lk_available", "LK available"), ("fb_1px", "Forward--backward"),
                    ("ncc_0.8", "Starting-patch NCC"), ("both", "Both")):
    records = [tapvid["sequences"][name]["counts"][gate] for name in sorted(tapvid["sequences"])]
    precision = np.array([c["accepted_correct"]/c["accepted"] for c in records])
    recall = np.array([c["accepted_correct"]/c["visible"] for c in records])
    assert f"{label} & {precision.mean()*100:.1f} & {recall.mean()*100:.1f}" in TEX
    if gate == "both":
        assert sum(c["planned"] for c in records) == 26374
        assert sum(c["accepted"] for c in records) == 4245
        assert sum(c["accepted_correct"] for c in records) == 4133
        assert sum(c["accepted_wrong"] for c in records) == 97
        assert sum(c["accepted_occluded"] for c in records) == 15
        indices = np.random.default_rng(677).integers(0, 20, (20000, 20))
        for values in (precision, recall):
            lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])*100
            assert f"{lo:.1f}--{hi:.1f}" in TEX

print(
    "Selected manuscript counts match retained records; human validation and general claims are not certified"
)
