import json
from pathlib import Path

import pytest

from paper.probe_movi_scene_calibration import order_statistic, run, scene_scores


def test_scene_sample_size_is_not_point_sample_size():
    assert order_statistic([1,2,3,4],.8)['multiplier'] == 4
    assert order_statistic([1,2,3,4],.95)['status'] == 'unbounded'
    assert order_statistic(list(range(19)),.95)['rank'] == 19
    assert order_statistic(list(range(19)),.95)['multiplier'] == 18
    row = dict(sequence='scene',eligible=True,error=2.,cycle=.5)
    assert scene_scores([row]*100,'cycle_scaled')[0]['score'] == 2
    with pytest.raises(ValueError,match='cannot silently omit'):
        scene_scores([dict(row,error=None)],'constant')


def test_retained_scene_calibration_replays():
    root=Path(__file__).resolve().parents[1]/'paper/results'
    result=run(root)
    assert result == json.loads((root/'movi-replication-shard01-v1/scene-calibration-v1.json').read_text())
    for method in result['methods'].values():
        assert len(method['calibration']) == 4
        for level in method['levels']:
            assert level['scenes'] == 4
            assert len(level['replication']) == 4
            for scene in level['replication']:
                assert scene['eligible'] == scene['labeled']+scene['unlabeled_eligible']
