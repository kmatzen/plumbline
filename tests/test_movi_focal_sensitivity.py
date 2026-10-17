import copy

import numpy as np

from paper.probe_movi_focal_sensitivity import probe
from paper.tapvid_camera_markers import project


def test_parallel_translation_equivalence_and_no_later_observation_leakage():
    poses = np.tile(np.eye(4), (24, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 1, 24)
    k = np.tile([[1., 0, .5], [0, 1., .5], [0, 0, 1.]], (24, 1, 1))
    xy = [project(np.array([.2, .3, 4]), p, c, np.array([256, 256])).tolist()
          for p, c in zip(poses, k)]
    track = dict(query_frame=0, query_xy=xy[0], records=[dict(frame=i, xy=xy[i]) for i in range(1, 24)])
    result = probe(track, poses, k)
    assert all(r['both_prefix_max_error_at_most_2px'] for r in result)
    assert max(v for r in result for v in r['forecast_difference_pixels']) < 1e-8
    changed = copy.deepcopy(track)
    for row in changed['records']:
        if row['frame'] >= 10:
            row['xy'] = [999, -999]
    assert probe(changed, poses, k) == result
    changed['records'][0]['xy'] = None
    missing = probe(changed, poses, k)
    assert all(r['fit']['status'] == 'missing_prefix_observations' for r in missing)
    assert all(r['forecasts'] == [None]*3 for r in missing)

    # Later forward motion breaks the parallel-camera focal/depth equivalence.
    forward = poses.copy()
    forward[10:, 2, 3] = np.linspace(.1, 1.8, 14)
    divergent = probe(track, forward, k)
    assert all(r['both_prefix_max_error_at_most_2px'] for r in divergent)
    assert max(v for r in divergent for v in r['forecast_difference_pixels'] if v is not None) > 8
