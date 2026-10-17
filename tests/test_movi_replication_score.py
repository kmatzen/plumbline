import json
from pathlib import Path

from paper.score_movi_replication import ARMS, accepts, pixel_label, predicted_label, run


def test_fixed_labels_and_reference_independent_selection():
    assert pixel_label(None) == 'unresolved'
    assert pixel_label(2) == 'close'
    assert pixel_label(8) == 'far'
    assert pixel_label(3) == 'borderline'
    assert predicted_label(3) == 'uncertain'
    row = dict(predicted_visible=True,motion_predicted='compatible',cycle_predicted='consistent')
    before = [accepts(row,arm) for arm in ARMS]
    row.update(reference='moving_foreground',tracking_error_pixels=1000)
    assert before == [accepts(row,arm) for arm in ARMS] == [True]*3
    row['predicted_visible'] = False
    assert not any(accepts(row,arm) for arm in ARMS)


def test_full_replication_score_replay_and_accounting():
    root = Path(__file__).resolve().parents[1]/'paper/results/movi-replication-shard01-v1'
    result = run(root)
    assert result == json.loads((root/'score-v1.json').read_text())
    assert len(result['points']) == 768
    assert len(result['placements']) == len({r['id'] for r in result['placements']}) == 1536
    assert sum(sum(c.values()) for c in result['tracking_confusion'].values()) == 768
    for arm in ARMS:
        score = result['placement_results'][arm]
        assert sum(score['answers'].values()) == 1536
        assert sum(sum(c.values()) for c in score['confusion'].values()) == 1536
    for point in result['points']:
        assert int(point['retained']['tracking_motion_cycle']) <= int(point['retained']['tracking_motion']) <= int(point['retained']['tracking'])
    for scene in result['paired_placement'].values():
        for arm in scene['variants'].values():
            assert arm['accepted'] == arm['reference_scored']+arm['reference_unresolved']
            assert arm['matched_reference']['count'] == arm['matched_estimate']['count']
