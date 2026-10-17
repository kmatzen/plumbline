import copy

import numpy as np

from paper.tapvid_camera_markers import label_proposals, project, proposals


def fixture():
    size = np.array([640, 480])
    poses = np.repeat(np.eye(4)[None], 70, axis=0)
    poses[:, 0, 3] = np.linspace(0, .5, 70)
    k = np.repeat(np.array([[.8, 0, .5], [0, 1., .5], [0, 0, 1.]])[None], 70, axis=0)
    point = np.array([0., 0., 4.])
    xy = [project(point, p, ki, size) for p, ki in zip(poses, k)]
    track = dict(query_frame=2, query_xy=xy[2].tolist(),
                 records=[dict(frame=i, xy=xy[i].tolist() if i <= 11 else None,
                               scoring_only={"poison": True}) for i in range(3, 70)])
    return track, poses, k, size, np.array(xy)


def test_later_tracks_and_annotation_metadata_cannot_affect_geometry():
    track, poses, k, size, xy = fixture()
    fit, rows = proposals(track, poses, k, size)
    assert fit["fit_frames"] == list(range(2, 12))
    assert [r["frame"] for r in rows] == [17, 32, 62]
    assert all(r["status"] == "renderable" for r in rows)
    for row in rows:
        np.testing.assert_allclose(row["candidate_xy"], xy[row["frame"]], atol=1e-7)
    changed = copy.deepcopy(track)
    for r in changed["records"]:
        r["scoring_only"] = {"different": False}
        if r["frame"] > 11:
            r["xy"] = [1e6, -1e6]
    assert proposals(changed, poses, k, size) == (fit, rows)


def test_labels_are_separate_and_keep_borderline_and_occlusion():
    track, poses, k, size, xy = fixture()
    _, rows = proposals(track, poses, k, size)
    frozen = copy.deepcopy(rows)
    truth = xy/size
    occ = np.zeros(70, bool)
    truth[32, 0] += 4/size[0]
    occ[62] = True
    labels = label_proposals(rows, truth, occ, size)
    assert [r["label"] for r in labels] == ["close", None, "occluded"]
    assert rows == frozen
    truth[17, 0] += 20/size[0]
    assert label_proposals(rows, truth, occ, size)[0]["label"] == "far"
    assert rows == frozen


def test_missing_prefix_keeps_all_planned_queries():
    track, poses, k, size, _ = fixture()
    track["records"][0]["xy"] = None
    _, rows = proposals(track, poses, k, size)
    assert len(rows) == 3 and all(r["status"] == "missing_prefix_observations" for r in rows)
    _, rows = proposals(dict(track=4), poses, k, size)
    assert len(rows) == 3 and all(r["status"] == "no_visible_query" for r in rows)
