import numpy as np

from paper.build_tapvid_judge_corpus import partitions, render, target_label


def test_label_margin_visibility_and_native_coordinates():
    assert target_label([12, 20], [10, 20], False) == "close"
    assert target_label([15, 20], [10, 20], False) is None
    assert target_label([18, 20], [10, 20], False) == "far"
    assert target_label([10, 20], [10, 20], True) == "occluded"


def test_video_partitions_are_stable_and_disjoint():
    names = [f"v-{i}" for i in range(10)]
    a = partitions(names)
    assert a == partitions(names[::-1])
    assert [list(a.values()).count(p) for p in ("train", "validation", "calibration")] == [6, 2, 2]


def test_renderer_has_no_label_inputs_and_preserves_native_crop_centers():
    a = np.zeros((200, 300, 3), np.uint8)
    b = a.copy()
    a[100, 150] = [1, 2, 3]
    b[90, 140] = [4, 5, 6]
    image = render(a, b, [150, 100], [140, 90])
    assert image.shape == (383, 544, 3)
    np.testing.assert_array_equal(image[294, 131], [1, 2, 3])
    np.testing.assert_array_equal(image[294, 411], [4, 5, 6])
