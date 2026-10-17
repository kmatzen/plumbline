import json

import pytest

from paper.evaluate_tapvid_lora import digest, select_camera_evaluation


def corpus(tmp_path):
    image = tmp_path/"panel.png"
    image.write_bytes(b"hash-only fixture")
    records = [dict(id=f"{m}:{s}:{f}", model=m, slot=s, frame=f, status="missing_prefix_observations")
               for m in ("dage", "da3") for s in range(12) for f in (30, 60, 89)]
    records[0].update(status="renderable", image=image.name, image_sha256=digest(image))
    export = tmp_path/"inference.jsonl"
    export.write_text(json.dumps(dict(id=records[0]["id"], image=image.name, prompt="frozen"))+"\n")
    manifest = dict(records=records, planned_presentations=72, prompt="frozen", export_sha256=digest(export))
    return manifest, dict(training_ids=[]), dict(prompt="frozen")


def test_unlabeled_application_keeps_unavailable_roster(tmp_path):
    manifest, train, dev = corpus(tmp_path)
    _, rows, allowed = select_camera_evaluation(tmp_path, train, manifest, dev)
    assert len(manifest["records"]) == 72 and len(rows) == len(allowed) == 1
    assert "target" not in rows[0]
    manifest["records"].pop()
    with pytest.raises(ValueError, match="roster"):
        select_camera_evaluation(tmp_path, train, manifest, dev)


def test_targets_or_prompt_changes_rejected(tmp_path):
    manifest, train, dev = corpus(tmp_path)
    export = tmp_path/"inference.jsonl"
    row = json.loads(export.read_text())
    row["target"] = "close"
    export.write_text(json.dumps(row)+"\n")
    manifest["export_sha256"] = digest(export)
    with pytest.raises(ValueError, match="target"):
        select_camera_evaluation(tmp_path, train, manifest, dev)
    dev["prompt"] = "modified"
    with pytest.raises(ValueError, match="prompt"):
        select_camera_evaluation(tmp_path, train, manifest, dev)
