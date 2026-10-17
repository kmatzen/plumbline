import numpy as np
import pytest

from paper.audit_movi_camera_metadata import project_centers


def test_camera_axes_translation_and_focal_ratio():
    points = np.array([[[2.,3.,-2.]],[[3.,4.,-2.]]])
    result = project_centers(points,np.array([[2.,3.,0.]]),np.array([[1.,0.,0.,0.]]),1.)
    np.testing.assert_allclose(result[:,0],[[.5,.5],[1.,0.]])


def test_camera_world_rotation_and_quaternion_sign():
    q = np.array([[np.sqrt(.5),0.,np.sqrt(.5),0.]])
    points = np.array([[[-2.,0.,0.]]])
    a = project_centers(points,np.zeros((1,3)),q,1.)
    np.testing.assert_allclose(a,[[[.5,.5]]],atol=1e-12)
    np.testing.assert_allclose(a,project_centers(points,np.zeros((1,3)),-q,1.))
    with pytest.raises(ValueError):
        project_centers(points,np.zeros((1,3)),q*2,1.)
