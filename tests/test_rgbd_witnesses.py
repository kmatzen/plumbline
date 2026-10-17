import json

import cv2
import numpy as np
import pytest

from paper.render_rgbd_witnesses import render
from paper.stage_rgbd_validation import sha


def test_witness_packet_checks_inputs_and_preserves_selection(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    depth_root = tmp_path / "depth"
    depth_root.mkdir()
    rgb = source / "frame.png"
    cv2.imwrite(str(rgb), np.full((480, 640, 3), 128, np.uint8))
    depth = depth_root / "depth.png"
    cv2.imwrite(str(depth), np.full((480, 640), 10000, np.uint16))
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"frames": [{"path": rgb.name, "file_sha256": sha(rgb)}] * 2}))
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps({"tracks": [[{"xy": [200, 200]}] * 2]}))
    depth_manifest = depth_root / "depth-manifest.json"
    depth_manifest.write_text(json.dumps({"rows": [
        {"status": "no_depth_within_20ms"},
        {"status": "staged", "depth_sha256": sha(depth),
         "nearest_depth": {"path": depth.name, "delta_seconds": .001}},
    ]}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "source_manifest_sha256": sha(manifest), "tracks_sha256": sha(tracks),
        "depth_manifest_sha256": sha(depth_manifest), "tracks": [
            {"slot": 0, "observed_spread": {"diameter": {"frames": [0, 1], "distance_m": .1}}},
            {"slot": 1, "observed_spread": {"diameter": None}},
        ],
    }))
    output = tmp_path / "packet"
    render(run, source, tracks, audit, depth_root, output)
    packet = json.loads((output / "manifest.json").read_text())
    assert len(packet["items"]) == 1
    assert packet["items"][0]["frames"] == [0, 0, 1]
    assert packet["items"][0]["sha256"] == sha(output / "slot-0.png")
    with pytest.raises(ValueError, match="preserve"):
        render(run, source, tracks, audit, depth_root, output)
    rgb.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        render(run, source, tracks, audit, depth_root, tmp_path / "changed")
