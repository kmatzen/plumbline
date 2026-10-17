import json

from paper.probe_static_projection_range import run
from paper.stage_static_rgbd_control import digest


def test_query_image_coordinates_and_sensor_depth_do_not_change_bounds(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"frames": [
        {"reference": {"matches": [{"values": [i*.01, 0, 0, 0, 0, 0, 1]}]}}
        for i in range(90)]}))
    control = tmp_path / "control.json"
    track = [{"xy": [320.1 - 535.4*i*.01/3, 247.6]} for i in range(90)]
    data = {"source_manifest_sha256": digest(source), "tracks": [track]}
    control.write_text(json.dumps(data))
    before = run(source, control, tmp_path / "before.json")
    for i in range(11, 90):
        track[i]["xy"] = None if i % 2 else [-1e9, 1e9]
    data["depth_tracks"] = "deliberately unusable sensor data"
    control.write_text(json.dumps(data))
    after = run(source, control, tmp_path / "after.json")
    assert before["rows"] == after["rows"]
