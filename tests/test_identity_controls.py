import cv2
import pytest
from paper.build_identity_controls import build


def test_controls_have_distinct_marked_targets_and_preserve_attempts(tmp_path):
    path = tmp_path / "controls"
    result = build(path)
    assert [r["target_shape"] for r in result["cases"]] == ["triangle", "rectangle", "circle"]
    assert len({r["image_sha256"] for r in result["cases"]}) == 3
    assert all(cv2.imread(str(path / r["name"])).shape == (550, 960, 3) for r in result["cases"])
    with pytest.raises(ValueError, match="preserve"):
        build(path)
