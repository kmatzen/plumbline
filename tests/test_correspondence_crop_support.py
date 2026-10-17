from paper.audit_correspondence_crop_support import in_crop


def test_crop_support_uses_rendered_rounded_center_and_both_axes():
    assert in_crop([180,20],[100.4,100.4])
    assert not in_crop([180.01,20],[100.4,100.4])
    assert not in_crop([100,19.99],[100.4,100.4])
