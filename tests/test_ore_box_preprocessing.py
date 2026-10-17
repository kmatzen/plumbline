import numpy as np
from paper.audit_ore_box_preprocessing import trace_box


def test_zero_distortion_roundtrip_and_scaling():
    row = dict(frame_number=0, x=60.0, y=90.0, width=120.0, height=150.0)
    result = trace_box(row, np.array([300, 300, 320, 240, 0, 0, 0, 0.0]))
    np.testing.assert_allclose(result["rescaled_xyxy"], [20, 30, 60, 80])
    np.testing.assert_allclose(result["roundtrip_error_pixels"], 0, atol=1e-10)
    assert not result["reversed_after_distortion"]


def test_raw_reversal_remains_visible():
    row = dict(frame_number=0, x=60.0, y=90.0, width=-30.0, height=150.0)
    result = trace_box(row, np.array([300, 300, 320, 240, 0, 0, 0, 0.0]))
    assert result["reversed_before_distortion"]
    assert result["reversed_after_integer_cast"]
