"""Scoring-only image displacement audit; not a reference-free input or DNF label."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from paper.probe_evidence_sensitivity import alternatives
from paper.render_layered_camera_control import K, anchors, camera_pose, project_anchors
from paper.stage_static_rgbd_control import digest


def project(cameras, points):
    homogeneous = np.c_[points, np.ones(len(points))]
    camera = np.einsum("tij,sj->tsi", cameras, homogeneous)[..., :3]
    if np.any(camera[..., 2] <= 0):
        raise ValueError("nonpositive reference-point depth")
    pixels = camera @ K.T
    return pixels[..., :2]/pixels[..., 2:]


def analyze(source):
    data = json.loads(gzip.decompress(source.read_bytes()))
    root = Path(__file__).parents[1]
    for name, expected in data["dependencies_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("control dependency changed")
    points, ids = anchors()
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    reference = project(cameras, points)
    visibility = np.stack([project_anchors(points, ids, t)[1] for t in range(24)])
    np.testing.assert_allclose(reference, data["reference_xy"], atol=1e-10)
    np.testing.assert_array_equal(visibility, data["reference_visible"])
    cases = []
    for name, changed in alternatives(cameras):
        world = points
        if name == "global_rigid_gauge":
            # The world points must change coordinates along with the cameras.
            gauge = np.linalg.inv(changed[0]) @ cameras[0]
            world = (np.c_[points, np.ones(len(points))] @ gauge.T)[:, :3]
        displacement = project(changed, world)-reference
        statuses = [[p for p in c["profiles"] if p["arm"] == "exact_visible"
                     and p["change"] == name][0]["slots"] for c in data["cases"]]
        if any([s["status"] for s in rows] != [s["status"] for s in statuses[0]]
               for rows in statuses[1:]):
            raise ValueError("oracle geometry should not depend on texture seed")
        slots = []
        for slot in range(12):
            indices = np.flatnonzero(visibility[12:, slot])+12
            delta = displacement[indices, slot]
            maximum = float(np.max(np.abs(delta))) if len(delta) else None
            slots.append(dict(slot=slot, surface_id=int(ids[slot]), query_frames=indices.tolist(),
                              displacement_xy=delta.tolist(), max_coordinate_displacement=maximum,
                              max_euclidean_displacement=float(np.linalg.norm(delta, axis=1).max()) if len(delta) else None,
                              oracle_status=statuses[0][slot]["status"]))
        cases.append(dict(change=name, slots=slots,
                          compatible_above_2px=sum(s["oracle_status"] == "jointly_compatible"
                              and s["max_coordinate_displacement"] is not None
                              and s["max_coordinate_displacement"] > 2+1e-8 for s in slots),
                          compatible_at_most_2px=sum(s["oracle_status"] == "jointly_compatible"
                              and s["max_coordinate_displacement"] is not None
                              and s["max_coordinate_displacement"] <= 2+1e-8 for s in slots)))
    return dict(scope="reference-only displacement audit, not DNF or a novel metric",
                interpretation="true 3D points fixed except global gauge; feasibility may refit unknown points; above-tolerance true-point displacement need not imply infeasibility",
                unit="12 unique geometric tracks, not 120 independent texture-seed tracks",
                source_sha256=digest(source), script_sha256=digest(Path(__file__)), cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.source)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k:v for k,v in c.items() if k != "slots"} for c in result["cases"]], indent=2))
