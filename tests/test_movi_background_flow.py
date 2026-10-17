import numpy as np

from paper.audit_movi_background_flow import sample_flow, compare


def test_linear_interpolation_and_no_extrapolation():
    y, x = np.mgrid[:5, :6]
    field = np.stack([2*x+y, x-3*y], axis=-1)
    result = sample_flow(field, np.array([[1.25, 2.5], [5, 4], [-.01, 0]]))
    np.testing.assert_allclose(result[:2], [[5, -6.25], [14, -7]])
    assert np.isnan(result[2]).all()


def test_temporal_pairing_and_rejected_cases_retained():
    xy = np.array([[[1., 1.], [2., 1.], [4., 1.]]])
    background = dict(projected_xy=xy, source_xy=xy[:, 0], in_image=np.ones((1, 3), bool),
        depth_rejected=np.array([[False, False, True]]), mask_rejected=np.zeros((1, 3), bool),
        matching_background_neighbors=np.full((1, 3), 4))
    encoded = np.zeros((3, 6, 6, 2), np.uint16)
    encoded[0, :, :, 1] = 100
    encoded[1, :, :, 1] = 200
    result = compare(background, encoded, np.array([0., 655.35]))
    np.testing.assert_allclose(result['flow_error_pixels'], 0, atol=1e-10)
    assert result['sampled'].tolist() == [[True, True]]
    assert result['depth_mask_interior_pair'].tolist() == [[True, False]]
