import numpy as np

from paper.audit_movi_marker_motion import join


def test_pixel_convention_background_and_uncertainty_retained():
    records = [dict(source_xy=[12, 12], frame=1, candidate_xy=[14, 13]),
               dict(source_xy=[44, 12], frame=1, candidate_xy=None)]
    temporal = dict(source_xy=np.array([[12.5, 12.5]]),
                    projected_xy=np.array([[[12.5, 12.5], [14.5, 13.5]]]),
                    world_displacement=np.array([[0, 1.]]),
                    counterfactual_static_error_pixels=np.array([[0, 2.]]))
    visibility = dict(supported=np.zeros((1, 2), bool), rejected=np.zeros((1, 2), bool))
    result = join(records, temporal, visibility)
    assert len(result) == 2
    assert result[0]['candidate_to_moving_surface_pixels'] == 0
    assert result[0]['visibility_evidence'] == 'unresolved'
    assert result[0]['world_displacement'] == 1
    assert result[1]['reference_status'] == 'no_foreground_reference'
    assert 'reference_status' not in records[0]
