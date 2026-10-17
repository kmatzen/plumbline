import json

import pytest
from paper.build_source_preview import build


@pytest.mark.parametrize(
    "frames,match",
    [
        ([{"timestamp": "1"}, {"timestamp": "1"}], "ordered"),
        ([{"timestamp": "nan"}, {"timestamp": "2"}], "finite"),
        ([{"timestamp": "1", "path": "../escape.png"}, {"timestamp": "2"}], "outside"),
        (
            [{"timestamp": "1", "path": "frame.png", "file_sha256": "wrong"}, {"timestamp": "2"}],
            "hash mismatch",
        ),
    ],
)
def test_invalid_inputs_do_not_create_preview(tmp_path, frames, match):
    (tmp_path / "frame.png").write_bytes(b"test")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": frames}))
    output = tmp_path / "preview"
    with pytest.raises(ValueError, match=match):
        build(manifest, tmp_path, output)
    assert not output.exists()
