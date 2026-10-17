"""Verify a matched presentation comparison; never score caption correctness."""

import argparse
import json
from pathlib import Path

from paper.stage_rgbd_validation import sha


def compare_pair(original, scaled, image_sha256):
    for row in (original, scaled):
        if row["status"] != "complete":
            raise ValueError("comparison requires completed attempts")
        if row["image_sha256"] != image_sha256:
            raise ValueError("image differs from declared input")
        audit = row["processed_input_audit"]
        if not audit["pixel_finite"] or not audit["placeholder_count_matches"]:
            raise ValueError("invalid processed input")
        if row["parse_error"] is not None or row["parsed_response"] != {"description": row["raw_response"]}:
            raise ValueError("unexpected caption record")
    keys = (
        "task", "image_sha256", "prompt_sha256", "checkpoint_files", "seed",
        "attention_helper_sha256", "attention_implementation_sha256",
        "quantization_config", "source_image_size", "torch", "transformers",
        "cuda", "gpu", "fp32_attention_scores", "bitsandbytes",
    )
    for key in keys:
        if original[key] != scaled[key]:
            raise ValueError(f"unmatched {key}")
    options = dict(scaled["settings"])
    if options.pop("min_pixels", None) != 576 * 28 * 28 or options != original["settings"]:
        raise ValueError("unexpected settings change")
    old = original["processed_input_audit"]
    new = scaled["processed_input_audit"]
    if old["actual_image_tokens"] != 36 or new["actual_image_tokens"] != 576:
        raise ValueError("unexpected realized image scale")
    return {
        "original_image_tokens": old["actual_image_tokens"],
        "scaled_image_tokens": new["actual_image_tokens"],
        "response_changed": original["raw_response"] != scaled["raw_response"],
        "original_response": original["raw_response"],
        "scaled_response": scaled["raw_response"],
        "visual_correctness": "not_assessed",
    }


def compare(original_root, scaled_root, output):
    if output.exists():
        raise ValueError("preserve earlier comparisons")
    manifest = json.loads((original_root / "input-manifest.json").read_text())
    rows = []
    for item in manifest["items"]:
        slot = item["slot"]
        old_path, new_path = (root / f"slot-{slot}.json" for root in (original_root, scaled_root))
        if not new_path.exists():
            rows.append({"slot": slot, "status": "not_retrieved"})
            continue
        old, new = (json.loads(p.read_text()) for p in (old_path, new_path))
        for root, record in ((original_root, old), (scaled_root, new)):
            if record["runner_sha256"] != sha(root / "executed-runner.py"):
                raise ValueError("executed runner mismatch")
            if record["attention_helper_sha256"] != sha(root / "executed-attention-helper.py"):
                raise ValueError("executed helper mismatch")
        rows.append({"slot": slot, "status": "matched", "original_sha256": sha(old_path),
                     "scaled_sha256": sha(new_path), **compare_pair(old, new, item["image_sha256"])})
    result = {"scope": "paired presentation audit, not visual accuracy or DNF validation",
              "input_manifest_sha256": sha(original_root / "input-manifest.json"),
              "script_sha256": sha(Path(__file__)), "rows": rows}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original_root", "scaled_root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    compare(args.original_root, args.scaled_root, args.output)
