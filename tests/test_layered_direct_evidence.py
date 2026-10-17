import numpy as np

from paper.probe_layered_direct_evidence import direct_tracks


def test_rejection_does_not_prevent_later_direct_match(monkeypatch):
    import paper.probe_layered_direct_evidence as module
    image = np.zeros((32, 32), dtype=np.uint8)
    answers = iter([False, True])
    def match(source, target, queries):
        assert source is image
        assert queries == [[15., 15.]]
        return [dict(accepted=next(answers), predicted_xy=[16., 15.], fb_pixels=0., ncc=1.)]
    monkeypatch.setattr(module, "match_source", match)
    tracks, records = direct_tracks([image]*3, [[15., 15.]])
    assert tracks == [[[15., 15.], None, [16., 15.]]]
    assert [r["frame"] for r in records] == [1, 2]
