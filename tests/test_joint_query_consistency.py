import numpy as np
import pytest

from paper.probe_joint_query_consistency import compare_queries


def setup_geometry():
    fit = np.repeat(np.eye(4)[None], 2, axis=0)
    query = fit.copy()
    query[:, 0, 3] = -1
    return fit, query, np.diag([100., 100., 1.])


def test_separate_feasible_points_do_not_imply_one_shared_point():
    fit, query, k = setup_geometry()
    result = compare_queries(fit, [[0, 0], [0, 0]], query, [[-10, 0], [-20, 0]], k, 1)
    assert result["individual_counts"] == {"compatible": 2}
    assert result["status"] == "joint_contradiction"


def test_common_witness_and_missing_coverage_are_separate():
    fit, query, k = setup_geometry()
    result = compare_queries(fit, [[0, 0], [0, 0]], query, [[-10, 0], None], k, 1)
    assert result["status"] == "jointly_compatible"
    assert result["planned_queries"] == 2
    assert result["observed_queries"] == 1
    assert result["observed_query_indices"] == [0]
    point = np.r_[result["combined"]["point_world"], 1.]
    projected = k @ query[0, :3] @ point
    assert np.max(np.abs(projected[:2] / projected[2] - [-10, 0])) <= 1 + 1e-5
    result = compare_queries(fit, [[0, 0], [0, 0]], query, [None, None], k, 1)
    assert result["status"] == "missing_query_observations"


def test_fitting_failure_not_attributed_to_query():
    fit, query, k = setup_geometry()
    result = compare_queries(fit, [[0, 0], [10, 0]], query, [[-10, 0], [-20, 0]], k, 1)
    assert result["status"] == "fitting_geometry_not_established"
    result = compare_queries(fit, [None, [0, 0]], query, [[-10, 0], [-20, 0]], k, 1)
    assert result["status"] == "insufficient_fitting_observations"


def test_query_order_does_not_change_joint_result():
    fit, query, k = setup_geometry()
    xy = [[-10, 0], [-20, 0]]
    a = compare_queries(fit, [[0, 0], [0, 0]], query, xy, k, 1)
    b = compare_queries(fit, [[0, 0], [0, 0]], query[::-1], xy[::-1], k, 1)
    assert a["status"] == b["status"]
    with pytest.raises(ValueError):
        compare_queries(fit, [[0, 0], [0, 0]], query[:0], [], k, 1)


def test_middle_error_is_not_hidden_by_a_compatible_endpoint():
    fit, query, k = setup_geometry()
    fit[1, 0, 3] = -.5
    result = compare_queries(fit, [[0, 0], [-10, 0]], query,
                             [[40, 0], [-20, 0]], k, 1)
    assert result["individual"][-1]["status"] == "compatible"
    assert result["individual"][0]["status"] == "query_contradiction"
    assert result["status"] == "joint_contradiction"
