import numpy as np
import pytest
from paper.audit_ore_semantics import cases
from paper.run_ore_demo import run, serializable

from plumbline.object_reprojection import score_static_track


def test_audit_controls_keep_geometric_failures_distinct():
    records = {}
    for name, poses, k, boxes in cases():
        records[name] = score_static_track(
            poses, k, boxes, (200, 200), [1, 5, 10], static_verified=True
        )
    assert records["exact_center_ray"]["score"] == 0
    assert records["behind_camera_projection"]["status"] == "unavailable"
    assert records["image_border_suppression"]["score"] > 0
    assert records["large_error_clipping"]["score"] > 1
    assert records["off_axis_radial_vs_axial_depth"]["score"] == 0


def test_demo_preserves_existing_outputs(tmp_path):
    with pytest.raises(ValueError, match="preserve"):
        run(tmp_path, tmp_path)


def test_demo_serialization_retains_numpy_values():
    assert serializable({"value": np.float64(0.25), "scale": np.array([5.0])}) == {
        "value": 0.25,
        "scale": [5.0],
    }
