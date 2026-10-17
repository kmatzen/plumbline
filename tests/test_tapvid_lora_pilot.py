import pytest
import torch
import json

from paper.train_tapvid_lora_pilot import digest, mask_prompt, select_training, memorization_schedule, repeat_schedule
from paper.evaluate_tapvid_lora import parse_label, select_evaluation


def test_repeated_full_study_preserves_first_pass_and_every_example():
    from collections import Counter
    rows = [dict(id=str(i)) for i in range(188)]
    result = repeat_schedule(rows, 20)
    assert len(result) == 3760
    assert result[:188] == rows
    assert result == repeat_schedule(rows, 20)
    assert result[188:376] != rows
    assert Counter(r["id"] for r in result) == {str(i): 20 for i in range(188)}
    for offset in range(0, len(result), 188):
        assert {r["id"] for r in result[offset:offset+188]} == {str(i) for i in range(188)}
    assert repeat_schedule(rows, 1) == rows
    with pytest.raises(ValueError, match="fixed"):
        repeat_schedule(rows, 2)


def test_memorization_is_balanced_fixed_and_independent_of_input_order():
    from collections import Counter
    rows = [dict(id=f"{label}-{i}", target=label) for label in ("close", "far", "occluded") for i in range(5)]
    selected, schedule = memorization_schedule(rows, 20)
    assert len(selected) == 9 and len(schedule) == 180
    assert Counter(r["target"] for r in selected) == dict(close=3, far=3, occluded=3)
    assert all(n == 20 for n in Counter(r["id"] for r in schedule).values())
    assert memorization_schedule(list(reversed(rows)), 20) == (selected, schedule)
    for offset in range(0, 180, 9):
        assert {r["id"] for r in schedule[offset:offset+9]} == {r["id"] for r in selected}
    with pytest.raises(ValueError, match="fixed"):
        memorization_schedule(rows, 21)
    with pytest.raises(ValueError, match="three training"):
        memorization_schedule(rows[:2], 20)


def test_training_fit_reads_only_exact_training_partition(tmp_path):
    image = tmp_path / "panel.png"
    image.write_bytes(b"hash checked input")
    row = dict(id="t", image=image.name, target="close", prompt="inspect")
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row)+"\n")
    record = dict(id="t", partition="train", status="labeled", label="close",
                  image=image.name, image_sha256=digest(image))
    manifest = dict(records=[record], prompt="inspect", exports_sha256={"train": digest(source)})
    train = dict(training_ids=["t"])
    assert select_evaluation(tmp_path, train, manifest, "train")[1] == [row]
    assert not (tmp_path / "validation.jsonl").exists()
    assert not (tmp_path / "calibration.jsonl").exists()
    with pytest.raises(ValueError, match="complete training partition"):
        select_evaluation(tmp_path, dict(training_ids=[]), manifest, "train")
    with pytest.raises(ValueError, match="only training-fit"):
        select_evaluation(tmp_path, train, manifest, "calibration")
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        select_evaluation(tmp_path, train, manifest, "train")


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
