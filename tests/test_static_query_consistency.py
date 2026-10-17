import numpy as np

from paper.probe_static_query_consistency import compare_query


def test_distinguishes_missing_fit_missing_query_and_query_contradiction():
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)
    cameras[1, 0, 3] = -1
    query = np.eye(4)
    query[0, 3] = -2
    k = np.diag([100., 100., 1.])
    xy = [[0, 0], [-20, 0]]
    good = compare_query(cameras, xy, query, [-40, 0], k, 1)
    bad = compare_query(cameras, xy, query, [40, 0], k, 1)
    assert good["status"] == "compatible"
    assert bad["status"] == "query_contradiction"
    assert good["early"] == bad["early"]
    assert compare_query(cameras, xy, query, None, k, 1)["status"] == "missing_query_observation"
    assert compare_query(cameras, [None, xy[1]], query, [-40, 0], k, 1)["status"] == "insufficient_fitting_observations"
    cameras[:] = np.eye(4)
    assert compare_query(cameras, xy, query, [40, 0], k, 1)["status"] == "fitting_geometry_not_established"
