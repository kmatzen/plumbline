import itertools

import numpy as np

from paper.audit_movi_visibility import evidence, normal_evidence


def test_normal_rotates_with_object_not_camera_position():
    quaternions = np.array([[[1.,0.,0.,0.],[0.,0.,1.,0.]]])
    world = np.array([[[0.,0.,-2.],[0.,0.,-2.]]])
    dots, front = normal_evidence(np.array([[0.,0.,1.]]),quaternions,world,np.zeros((2,3)))
    np.testing.assert_allclose(dots,[[-2.,2.]])
    assert front.tolist() == [True]
    np.testing.assert_allclose(dots,normal_evidence(np.array([[0.,0.,1.]]),-quaternions,world,np.zeros((2,3)))[0])


def fixture():
    cube = np.array(list(itertools.product((-1.,1.),repeat=3)))+[0.,0.,-2.]
    a = dict(object_coordinates=np.full((1,8,8,3),32768,dtype=np.uint16),
             instances_bboxes_3d=cube[None,None],instances_quaternions=np.array([[[1.,0.,0.,0.]]]),
             camera_positions=np.zeros((1,3)))
    normals = np.tile(np.array([32768,32768,65535],dtype=np.uint16),(1,8,8,1))
    temporal = dict(source_xy=np.array([[4.5,4.5]]),instance=np.array([1]),in_image=np.array([[True]]),
                    neighbor_depth_max=np.array([[2.]]),radial_depth=np.array([[2.]]),matching_neighbor_count=np.array([[4]]))
    return a,normals,temporal


def test_untrusted_normals_abstain_unless_other_evidence_rejects():
    a,n,t = fixture()
    assert evidence(a,n,t)['supported'].item()
    for y,x in ((3,3),(3,5),(5,3),(5,5)):
        n[0,y,x]=[65535,0,0]
    result = evidence(a,n,t)
    assert result['unresolved'].item() and not result['supported'].item()
    t['matching_neighbor_count'][:]=0
    result = evidence(a,n,t)
    assert result['rejected'].item() and not result['unresolved'].item()


def test_depth_and_offscreen_rejection_keep_all_slots():
    a,n,t = fixture()
    t['neighbor_depth_max'][:]=1.
    assert evidence(a,n,t)['depth_rejected'].item()
    t['in_image'][:]=False
    t['neighbor_depth_max'][:]=np.nan
    result=evidence(a,n,t)
    assert result['outside_or_behind'].item()
    assert result['rejected'].size == 1
    assert not result['depth_rejected'].item()
