import numpy as np

from paper.audit_movi_background import unproject, background_arrays
from paper.audit_movi_camera_metadata import project_centers


def test_off_axis_radial_depth_and_rotation_roundtrip():
    xy = np.array([[12., 44.], [236, 12]])
    q = np.array([np.sqrt(.5), 0, np.sqrt(.5), 0])
    position = np.array([3., 4, 5])
    world = unproject(xy, np.array([2., 7.]), position, q, 1., [256, 256])
    np.testing.assert_allclose(np.linalg.norm(world-position, axis=1), [2, 7])
    uv = project_centers(world[:, None], position[None], q[None], 1.)
    np.testing.assert_allclose(uv[:, 0]*256-.5, xy, atol=1e-12)


def test_static_camera_background_occlusion_is_not_removed():
    masks = np.zeros((24, 256, 256), np.uint8)
    masks[5, :, :] = 1
    a = dict(segmentations=masks, metadata_depth_range=np.array([1., 3.]),
             depth=np.full(masks.shape, 32768, np.uint16),
             camera_focal_length=np.array([1.]), camera_sensor_width=np.array([1.]),
             camera_positions=np.zeros((24, 3)), camera_quaternions=np.tile([1., 0, 0, 0], (24, 1)))
    result = background_arrays(a, np.zeros((256, 256, 2), np.uint16), np.array([0., 1.]))
    assert len(result['slots']) == 64
    assert result['projected_xy'].shape == (64, 24, 2)
    assert result['mask_rejected'][:, 5].all()
    assert result['flow_error_pixels'].max() < 1e-10


def test_flow_row_column_order_and_quantization():
    masks = np.zeros((24, 256, 256), np.uint8)
    depth = np.full(masks.shape, 32768, np.uint16)
    positions = np.zeros((24, 3))
    positions[1:, 0] = .1
    a = dict(segmentations=masks, metadata_depth_range=np.array([1., 3.]), depth=depth,
             camera_focal_length=np.array([1.]), camera_sensor_width=np.array([1.]),
             camera_positions=positions, camera_quaternions=np.tile([1., 0, 0, 0], (24, 1)))
    y, x = np.mgrid[:256, :256]
    radial = 1+32768/65535*2
    forward_depth = radial/np.sqrt(1+((x+.5)/256-.5)**2+((y+.5)/256-.5)**2)
    delta_column = -.1/forward_depth*256
    raw_flow = np.stack([np.zeros_like(delta_column), delta_column], axis=-1)
    encoded = np.rint((raw_flow+20)/40*65535).astype(np.uint16)
    result = background_arrays(a, encoded, np.array([-20., 20.]))
    assert result['flow_error_pixels'].max() <= np.sqrt(2)*20/65535+1e-10
