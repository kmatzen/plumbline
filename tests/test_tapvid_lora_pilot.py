import pytest
import torch
import json

from paper.train_tapvid_lora_pilot import digest, mask_prompt, select_training
from paper.evaluate_tapvid_lora import parse_label


def test_only_assistant_extension_is_supervised():
    full = torch.tensor([[1, 4, 4, 4, 2, 10, 11]])
    result = mask_prompt(full, full[:, :5])
    assert result.tolist() == [[-100]*5+[10, 11]]
    assert full.tolist() == [[1, 4, 4, 4, 2, 10, 11]]
    with pytest.raises(ValueError):
        mask_prompt(full, torch.tensor([[1, 3]]))
    with pytest.raises(ValueError):
        mask_prompt(full, full)


def test_evaluation_parser_does_not_repair_unsupported_answers():
    assert parse_label(" Close\n") == "close"
    assert parse_label("uncertain") == "uncertain"
    assert parse_label("It is close") == "invalid"
    assert parse_label('```close```') == "invalid"
    assert parse_label("") == "invalid"


def test_training_membership_and_labels_checked_without_other_exports(tmp_path):
    records, rows = [], []
    for label in ("close", "far", "occluded"):
        image = tmp_path / (label+".png")
        image.write_bytes(label.encode())  # Membership test does not decode pixels.
        records.append(dict(id=label, partition="train", status="labeled", label=label,
                            image=image.name, image_sha256=digest(image)))
        rows.append(dict(id=label, image=image.name, target=label, prompt="inspect"))
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    manifest = dict(records=records, prompt="inspect", exports_sha256={"train": digest(path)})
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    assert len(select_training(tmp_path, 3)) == 3
    assert not (tmp_path / "calibration.jsonl").exists()
    manifest["records"][0]["partition"] = "calibration"
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership"):
        select_training(tmp_path, 3)
