import numpy as np

from paper.probe_consistency_attribution import assess, make_case, scene


def test_interventions_do_not_change_fitting_inputs():
    clean_cameras, clean_tracks = make_case(0, "clean")
    for condition in ("camera_yaw_1", "camera_yaw_5", "bad_track_1", "bad_track_3", "coherent_bad_tracks"):
        cameras, tracks = make_case(0, condition)
        np.testing.assert_array_equal(cameras[:12], clean_cameras[:12])
        np.testing.assert_array_equal(np.array(tracks)[:, :12], np.array(clean_tracks)[:, :12])
    cameras, tracks = make_case(0, "coherent_bad_tracks")
    np.testing.assert_array_equal(cameras, clean_cameras)
    assert not np.array_equal(tracks, clean_tracks)


def test_clean_and_missing_are_not_conflated():
    cameras, tracks = make_case(0, "clean")
    result = assess(cameras, tracks, 1)
    assert result["counts"] == {"jointly_compatible": 12}
    cameras, tracks = make_case(0, "missing_queries")
    result = assess(cameras, tracks, 1)
    assert result["counts"] == {"missing_query_observations": 12}
    assert result["unresolved_fraction_of_all_slots"] == 1


def test_single_track_intervention_does_not_modify_other_tracks():
    cameras, clean = make_case(0, "clean")
    changed_cameras, changed = make_case(0, "bad_track_1")
    np.testing.assert_array_equal(cameras, changed_cameras)
    np.testing.assert_array_equal(clean[1:], changed[1:])
    result = assess(changed_cameras, changed, 1)
    assert result["counts"] == {"joint_contradiction": 1, "jointly_compatible": 11}


def test_scene_generator_is_deterministic_without_global_rng_mutation():
    state = np.random.get_state()
    for a, b in zip(scene(9), scene(9)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
