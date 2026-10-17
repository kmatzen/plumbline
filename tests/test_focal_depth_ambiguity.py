from paper.probe_focal_depth_ambiguity import run


def test_correspondence_equivalent_models_have_different_metric_anchor_error():
    result = run()
    assert len(result['cases']) == 5
    for case in result['cases']:
        assert case['maximum_correspondence_difference_pixels'] < 1e-10
        assert case['maximum_later_forecast_error_pixels'] < 1e-8
        if case['focal_factor'] == 1:
            assert case['maximum_known_anchor_task_error_pixels'] == 0
        else:
            assert case['mean_known_anchor_task_error_pixels'] > 1
