import json

import pytest
from paper.build_reference_free_review import digest
from paper.build_track_review import render_review, revise_packet, select_slots


def test_revision_preserves_videos_and_records_origin(tmp_path):
    previous = tmp_path / "v1"
    previous.mkdir()
    (previous / "example-1.mp4").write_bytes(b"test video")
    (previous / "frames-1").mkdir()
    (previous / "frames-1/000000.png").write_bytes(b"test image")
    item = {"example": 1, "video": "example-1.mp4", "sha256": digest(previous / "example-1.mp4")}
    (previous / "study.json").write_text(json.dumps({"items": [item]}))
    output = tmp_path / "v2"
    revise_packet(previous, output)
    study = json.loads((output / "study.json").read_text())
    assert study["previous_study_sha256"] == digest(previous / "study.json")
    assert digest(output / "example-1.mp4") == item["sha256"]
    assert study["items"][0]["starting_image_sha256"] == digest(previous / "frames-1/000000.png")
    assert item["sha256"] in (output / "index.html").read_text()
    with pytest.raises(ValueError, match="preserve"):
        revise_packet(previous, output)
    (previous / "example-1.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        revise_packet(previous, tmp_path / "bad")


def test_review_requires_target_identity_before_tracking_and_preserves_missingness():
    page = render_review([{"example": 1, "video": "example-1.mp4"}])
    assert 'src="frames-1/000000.png"' in page
    assert "<fieldset disabled>" in page
    assert "No, I cannot identify it" in page
    assert "'not_assessed'" in page
    assert "frame.checkValidity()" in page
    assert "not camera DNF" in page
    assert "not sent anywhere or saved automatically" in page
    assert "fetch(" not in page


def test_review_escapes_video_paths():
    page = render_review([{"example": 1, "video": 'a" onerror="bad'}])
    assert 'src="a&quot; onerror=&quot;bad"' in page


def test_selection_depends_only_on_first_frame_not_later_survival():
    tracks = [
        [{"xy": None}],
        [{"xy": [1, 2]}, {"xy": None}],
        [{"xy": [3, 4]}, {"xy": [8, 9]}],
        [{"xy": [5, 6]}],
        [{"xy": [7, 8]}],
    ]
    assert select_slots(tracks) == [1, 2, 3]
    tracks[1][1]["xy"] = [100, 200]
    tracks[2][1]["xy"] = None
    assert select_slots(tracks) == [1, 2, 3]
