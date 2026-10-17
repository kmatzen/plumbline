import cv2
import numpy as np
import pytest

from paper.decode_movi_geometry_pilot import images


@pytest.mark.parametrize("dtype", [np.uint8,np.uint16])
def test_rgb_channel_order_preserves_integer_precision(dtype):
    rgb = np.zeros((3,4,3),dtype=dtype)
    rgb[...,0],rgb[...,1],rgb[...,2] = 11,22,200 if dtype == np.uint8 else 60000
    ok, encoded = cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    assert ok
    decoded = images([encoded.tobytes()],(1,3,4,3),dtype,rgb=True)
    np.testing.assert_array_equal(decoded[0],rgb)


def test_depth_shape_dtype_and_invalid_png_rejected():
    depth = np.full((3,4),60000,dtype=np.uint16)
    ok, encoded = cv2.imencode('.png',depth)
    assert ok
    np.testing.assert_array_equal(images([encoded.tobytes()],(1,3,4),np.uint16)[0],depth)
    for shape,dtype in (((1,4,3),np.uint16),((1,3,4),np.uint8)):
        with pytest.raises(ValueError):
            images([encoded.tobytes()],shape,dtype)
    with pytest.raises(ValueError):
        images([b'not a png'],(1,3,4),np.uint16)
