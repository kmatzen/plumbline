import itertools

import numpy as np

from paper.audit_movi_surfaces import audit_arrays, object_to_world


def test_ordered_box_transform():
    cube = np.array(list(itertools.product((0.,1.),repeat=3)))
    matrix = np.array([[0.,2.,0.],[-3.,0.,0.],[0.,0.,4.]])
    corners = cube@matrix + [7.,8.,9.]
    points = np.array([[.2,.4,.8],[0.,0.,0.],[1.,1.,1.]])
    np.testing.assert_allclose(object_to_world(points,corners),points@matrix+[7.,8.,9.])


def test_foreground_grid_keeps_boundaries_and_validates_radial_depth():
    cube = np.array(list(itertools.product((-1.,1.),repeat=3)))
    corners = cube+np.array([0.,0.,-3.])
    coords = np.zeros((1,256,256,3),dtype=np.uint16)
    # Surface at z=-2, x=1, y~0 projects to pixel center (256,128),
    # deliberately not at the chosen grid pixel (4.5,4.5).
    coords[0,4,4] = [65535,32768,65535]
    mask = np.zeros((1,256,256),dtype=np.uint8); mask[0,4,4]=1
    a = dict(segmentations=mask,object_coordinates=coords,instances_bboxes_3d=corners[None,None],
             camera_quaternions=np.array([[1.,0.,0.,0.]]),camera_positions=np.zeros((1,3)),
             camera_focal_length=np.array([1.]),camera_sensor_width=np.array([1.]),
             metadata_depth_range=np.array([1.,3.]),
             depth=np.full((1,256,256),round((np.sqrt(5)-1)/2*65535),dtype=np.uint16))
    result = audit_arrays(a)
    assert result['frame'].tolist() == [0]
    assert result['interior'].tolist() == [False]
    assert result['positive_depth'].tolist() == [True]
    assert result['projection_error_pixels'][0] > 170
    assert result['radial_depth_error'][0] < 2e-5
