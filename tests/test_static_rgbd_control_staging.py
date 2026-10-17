import io
import tarfile

import pytest

from paper.stage_static_rgbd_control import SEQUENCE, nearest, parse_index, safe_image_path, stage


def test_association_keeps_ambiguity_and_missingness_local():
    rows = [{"timestamp": t, "values": [str(i)], "line": i} for i, t in enumerate([0., 1., 1., 2.])]
    assert nearest(0., rows)["status"] == "associated"
    assert nearest(1., rows)["status"] == "ambiguous_timestamp"
    assert len(nearest(1., rows)["matches"]) == 2
    assert nearest(.5, rows)["matches"][0]["timestamp"] == 0
    assert nearest(.5, rows)["status"] == "outside_20ms"
    assert nearest(0., [])["status"] == "empty_index"
    for path in ("../a.png", "/rgb/a.png", "rgb/../a.png", "depth/a.png"):
        with pytest.raises(ValueError, match="unsafe"):
            safe_image_path(path, "rgb")


def test_fixed_rgb_selection_does_not_depend_on_reference_or_depth(tmp_path):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        def add(name, payload):
            info = tarfile.TarInfo(f"{SEQUENCE}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        add("rgb.txt", "".join(f"{i} rgb/{i}.png\n" for i in range(270)).encode())
        add("depth.txt", b"3 depth/3.png\n6 depth/6.png\n6 depth/other.png\n")
        add("groundtruth.txt", b"3 0 0 0 0 0 0 1\n")
        for i in range(0, 270, 3):
            add(f"rgb/{i}.png", f"synthetic-image-{i}".encode())
        add("depth/3.png", b"synthetic-depth")
    output = tmp_path / "selected"
    record = stage(archive, output)
    assert len(record["frames"]) == 90
    assert [r["rgb_index"] for r in record["frames"]] == list(range(0, 270, 3))
    assert record["frames"][0]["depth"]["status"] == "outside_20ms"
    assert record["frames"][1]["depth"]["status"] == "associated"
    assert record["frames"][2]["depth"]["status"] == "ambiguous_timestamp"
    assert record["association_counts"]["reference"] == {"outside_20ms": 89, "associated": 1}
    assert (output / "rgb/0.png").exists() and (output / "rgb/267.png").exists()
    assert not (output / "depth/6.png").exists()
    with pytest.raises(ValueError, match="preserve"):
        stage(archive, output)


@pytest.mark.parametrize("kind", ["duplicate", "symlink"])
def test_rejects_ambiguous_or_nonordinary_archive_members(tmp_path, kind):
    archive = tmp_path / "bad.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(f"{SEQUENCE}/rgb.txt")
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        else:
            tar.addfile(info, io.BytesIO(b""))
            tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(ValueError, match="duplicate|ordinary"):
        stage(archive, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_nonfinite_timestamp_is_not_associated():
    with pytest.raises(ValueError, match="invalid timestamp"):
        parse_index(b"nan rgb/a.png\n", 2)
