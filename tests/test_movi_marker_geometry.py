import copy

import numpy as np

from paper.build_movi_marker_geometry import marker_rows, track_rgb


def test_fixed_slots_and_lost_tracks_retained():
    tracks = track_rgb(np.zeros((24, 256, 256, 3), np.uint8))
    assert len(tracks) == 64
    assert tracks[0]['query_xy'] == [12, 12]
    assert tracks[-1]['query_xy'] == [236, 236]
    assert all(len(t['records']) == 23 for t in tracks)
    assert all(r['xy'] is None for t in tracks for r in t['records'])


def test_later_observations_cannot_change_geometry():
    poses = np.tile(np.eye(4), (24, 1, 1))
    poses[:, 0, 3] = np.arange(24)*.01
    calibration = np.tile(np.array([[1., 0, .5], [0, 1, .5], [0, 0, 1]]), (24, 1, 1))
    track = dict(query_frame=0, query_xy=[127.5, 127.5], records=[
        dict(frame=i, xy=[127.5-i*.01/3*256, 127.5]) for i in range(1, 24)])
    changed = copy.deepcopy(track)
    for row in changed['records'][9:]:
        row['xy'] = None
    before = marker_rows(track, poses, calibration)
    assert before == marker_rows(changed, poses, calibration)
    assert before[0]['status'] == 'fitted'
    assert [r['frame'] for r in before[1]] == [12, 18, 23]
