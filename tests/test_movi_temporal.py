import itertools

import numpy as np

from paper.audit_movi_temporal import temporal_arrays


def fixture():
    corners = np.array(list(itertools.product((-1.,1.),repeat=3)))+[0.,0.,-2.]
    boxes = np.stack([corners,corners])[None]
    mask = np.zeros((2,256,256),dtype=np.uint8); mask[:,4,4]=1
    coords = np.zeros((2,256,256,3),dtype=np.uint16)
    coords[0,4,4]=np.round(np.array([.017578125,.982421875,.5])*65535).astype(np.uint16)
    return dict(segmentations=mask,object_coordinates=coords,instances_bboxes_3d=boxes,
                camera_positions=np.zeros((2,3)),camera_quaternions=np.tile([1.,0.,0.,0.],(2,1)),
                camera_focal_length=np.array([1.]),camera_sensor_width=np.array([1.]),
                metadata_depth_range=np.array([1.,4.]),depth=np.full((2,256,256),32768,dtype=np.uint16),
                initial_is_dynamic=np.array([0]))


def test_initial_flag_does_not_override_released_motion():
    data = fixture()
    data['instances_bboxes_3d'][0,1,:,0] += .25
    result = temporal_arrays(data)
    assert result['initial_is_dynamic'].tolist() == [0]
    np.testing.assert_allclose(result['world_displacement'],[[0,.25]])
    np.testing.assert_allclose(result['counterfactual_static_error_pixels'],[[0,32]],atol=.001)
    assert result['matching_neighbor_count'][0,1] == 0
    assert result['in_image'].all()


def test_camera_motion_is_not_object_motion_and_offscreen_is_retained():
    data = fixture()
    data['camera_positions'][1,0] = 20
    result = temporal_arrays(data)
    np.testing.assert_allclose(result['world_displacement'],0)
    np.testing.assert_allclose(result['counterfactual_static_error_pixels'],0)
    assert result['projected_xy'].shape == (1,2,2)
    assert not result['in_image'][0,1]
    assert np.isnan(result['neighbor_depth_min'][0,1])
    assert 'visible' not in result
