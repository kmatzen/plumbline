"""Account for every retained control attempt without repairing model answers."""

import argparse
import json
from pathlib import Path

from paper.build_reference_free_review import digest
from paper.run_vlm_dnf_pilot import parse_response


def wrapper_only_audit(raw):
    """Inspect one exact Markdown wrapper; do not change JSON fields or API acceptance."""
    stripped = raw.strip()
    for prefix in ("```json\n", "```\n"):
        if stripped.startswith(prefix) and stripped.endswith("\n```"):
            inner = stripped[len(prefix) : -4]
            if "```" not in inner:
                return parse_response(inner, "geometry")
    return None, "not_single_json_fence"


def assess(result, expected):
    if result["status"] != "complete":
        return {
            "execution": result["status"],
            "response_status": "unavailable",
            "verdict": None,
            "matches_control": None,
        }
    parsed, error = parse_response(result["raw_response"], "geometry")
    if parsed != result["parsed_response"] or error != result["parse_error"]:
        raise ValueError("retained parsing result differs from strict reparse")
    verdict = parsed["verdict"] if parsed else None
    audit, audit_error = wrapper_only_audit(result["raw_response"]) if error else (parsed, None)
    audit_verdict = audit["verdict"] if audit else None
    return {
        "execution": "complete",
        "response_status": error or "accepted",
        "verdict": verdict,
        "matches_control": verdict == expected
        if verdict in {"consistent", "inconsistent"}
        else None,
        "wrapper_only_audit": {
            "scope": "descriptive semantic audit only; original strict API acceptance unchanged",
            "error": audit_error,
            "verdict": audit_verdict,
            "matches_control": audit_verdict == expected
            if audit_verdict in {"consistent", "inconsistent"}
            else None,
        },
    }


def analyze(root, output):
    if output.exists():
        raise ValueError("preserve previous analyses")
    controls_path = root / "geometry-controls/controls.json"
    controls = json.loads(controls_path.read_text())
    by_hash = {c["image_sha256"]: c for c in controls["cases"]}
    paths = [("3B-fp16-1280", root / "geometry-controls/control_on-qwen3b.json")]
    for name in ("on", "off"):
        paths.extend(
            [
                ("3B-fp16-768", root / f"geometry-controls/control_{name}-qwen3b-768.json"),
                ("7B-NF4-fp16-scores-768", root / f"qwen7b-nf4-controls-v1/control_{name}.json"),
                (
                    "7B-NF4-fp32-scores-768",
                    root / f"qwen7b-nf4-controls-v1/fp32-control_{name}.json",
                ),
            ]
        )
    rows = []
    for configuration, path in paths:
        result = json.loads(path.read_text())
        control = by_hash[result["image_sha256"]]
        rows.append(
            {
                "configuration": configuration,
                "control": control["name"],
                "source": str(path),
                "source_sha256": digest(path),
                "expected_visual_relation": control["expected_visual_relation"],
                **assess(result, control["expected_visual_relation"]),
            }
        )
    data = {
        "scope": "two constructed development controls; no reliability, human agreement, or camera-DNF accuracy estimate",
        "script_sha256": digest(Path(__file__)),
        "controls_sha256": digest(controls_path),
        "policy": "strict parsing retained; exact single-fence semantic audit reported separately; no JSON field repair or promotion to accepted labels",
        "rows": rows,
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    for row in rows:
        print(
            row["configuration"],
            row["control"],
            row["execution"],
            row["response_status"],
            row["verdict"],
            row["matches_control"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
