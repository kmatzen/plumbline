import numpy as np

from paper.probe_layered_minimum_error import minimum_box_error


def test_coincident_camera_conflict_preserves_boundary_uncertainty():
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)
    result = minimum_box_error(cameras, np.array([[159.5, 119.5], [161.5, 119.5]]))
    # Inconsistent rays can meet at zero depth: the underlying solver reports
    # a boundary solution rather than infeasibility. Do not invent a bound.
    assert result["status"] == "numerically_unresolved"
    assert result["lower"] <= 1 <= result["upper"]


def test_bisection_brackets_known_monotone_oracle(monkeypatch):
    import paper.probe_layered_minimum_error as module
    def support(cameras, k, boxes):
        tolerance = (boxes[0, 2]-boxes[0, 0])/2
        return dict(status="positive_witness", point_world=[0, 0, 1]) if tolerance >= 1 else dict(status="solver_infeasible")
    monkeypatch.setattr(module, "box_support", support)
    result = minimum_box_error(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros((2, 2)))
    assert result["status"] == "numerical_bracket"
    assert result["lower"] <= 1 <= result["upper"]
    assert result["upper"]-result["lower"] <= .01


def test_numerical_uncertainty_does_not_become_infeasibility(monkeypatch):
    import paper.probe_layered_minimum_error as module
    answers = iter([dict(status="positive_witness", point_world=[0, 0, 1]),
                    dict(status="boundary_or_numerical")])
    monkeypatch.setattr(module, "box_support", lambda *args: next(answers))
    result = minimum_box_error(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros((2, 2)))
    assert result["status"] == "numerically_unresolved"
    assert result["lower"] == 0
    assert result["upper"] == 64
