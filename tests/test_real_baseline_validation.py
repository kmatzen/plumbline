import pytest

from paper.stage_static_rgbd_control import nearest
from paper.validate_real_camera_baseline import reference_poses


def test_association_is_verified_and_missing_reference_not_imputed():
    index = [dict(timestamp=1., values=[0, 0, 0, 0, 0, 0, 1])]
    manifest = dict(frames=[dict(timestamp=t, reference=nearest(t, index)) for t in (1., 2.)])
    poses, associations = reference_poses(manifest, index)
    assert poses[0].shape == (4, 4)
    assert poses[1] is None
    assert associations[1]["status"] == "outside_20ms"
    manifest["frames"][0]["reference"] = dict(status="associated", matches=[])
    with pytest.raises(ValueError, match="association"):
        reference_poses(manifest, index)
