import numpy as np

from paper.probe_real_evidence_sensitivity import K, alternatives, assess
from paper.probe_available_fit_consistency import compare_available


def inputs():
    poses = np.repeat(np.eye(4)[None], 90, axis=0)
    poses[:, 0, 3] = np.linspace(0, .5, 90)
    point = np.array([.2, .1, 4.])
    pixels = (point-poses[:, :3, 3]) @ K.T
    track = [{"xy": (p[:2]/p[2]).tolist()} for p in pixels]
    track[7]["xy"] = None
    return poses, track


def test_real_length_profile_matches_existing_kernel_and_preserves_prefix():
    poses, track = inputs()
    base = assess(np.linalg.inv(poses), track, 30, 2)["status"]
    for name, cameras in alternatives(poses, 30):
        actual = assess(cameras, track, 30, 2)["status"]
        expected = compare_available(cameras[1:31], [t["xy"] for t in track[1:31]],
                   cameras[31:], [t["xy"] for t in track[31:]], K, 2)["status"]
        assert actual == expected
        if name == "global_rigid_gauge":
            assert actual == base
        else:
            np.testing.assert_allclose(cameras[:31], np.linalg.inv(poses[:31]), atol=1e-14)
