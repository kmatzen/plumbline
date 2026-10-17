import hashlib
import io
import json
import tarfile

import pytest
from paper.stage_rgbd_validation import associate, stage


def test_association_retains_out_of_tolerance_and_uses_earlier_tie():
    assert associate(2, [(1, "a"), (3, "b")])["path"] == "a"
    assert not associate(2, [(1, "a")])["within_tolerance"]
    assert associate(2, []) is None


def test_staging_checks_rgb_identity_and_keeps_missing_rows(tmp_path):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in [
            ("seq/depth.txt", b"1.005 depth/a.png\n"),
            ("seq/depth/a.png", b"depth"),
            ("seq/rgb/a.png", b"rgb"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sequence": "seq",
                "frames": [
                    {
                        "path": "seq/rgb/a.png",
                        "timestamp": str(t),
                        "file_sha256": hashlib.sha256(b"rgb").hexdigest(),
                    }
                    for t in (1.0, 2.0)
                ],
            }
        )
    )
    output = tmp_path / "out"
    result = stage(archive, manifest, output)
    assert [r["status"] for r in result["rows"]] == ["staged", "no_depth_within_20ms"]
    assert (output / "depth/a.png").read_bytes() == b"depth"
    with pytest.raises(ValueError, match="preserve"):
        stage(archive, manifest, output)
    data = json.loads(manifest.read_text())
    data["frames"][0]["file_sha256"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs"):
        stage(archive, manifest, tmp_path / "bad")
