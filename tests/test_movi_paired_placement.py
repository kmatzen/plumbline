import json
from pathlib import Path

from paper.audit_movi_paired_placement import aggregate, preference, run


def test_empty_aggregate_is_not_a_perfect_score():
    assert aggregate([], 'errors')['mean'] is None
    assert preference(dict(dage=1., da3=2.)) == 'dage'
    assert preference(dict(dage=2., da3=1.)) == 'da3'
    assert preference(dict(dage=1., da3=1.)) == 'tie'


def test_exact_replay_and_nested_paired_support():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    result = run(root/'movi-camera-pilot-v1', root/'movi-temporal-verifier-v1', root/'movi-reverse-tracking-v1')
    assert result == json.loads((root/'movi-reverse-tracking-v1/paired-placement-v1.json').read_text())
    assert len(result['records']) == 768
    for scene in result['results'].values():
        v = scene['variants']
        assert v['tracking_motion_cycle']['accepted'] <= v['tracking_motion']['accepted'] <= v['tracking']['accepted'] <= scene['common_renderable'] <= scene['planned']
        for arm in v.values():
            assert arm['matched_estimate']['count'] == arm['matched_reference']['count'] == arm['reference_scored']
            assert arm['reference_scored']+arm['reference_unresolved'] == arm['accepted']
