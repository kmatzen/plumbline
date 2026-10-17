import json
from pathlib import Path

from paper.probe_movi_location_sensitivity import run


def test_shared_location_bounds_replay_and_contain_admissible_references():
    root = Path(__file__).resolve().parents[1]/'paper/results/movi-replication-shard01-v1'
    result = run(root)
    assert result == json.loads((root/'location-sensitivity-v1.json').read_text())
    assert len(result['cases']) == 8
    for case in result['cases']:
        previous = None
        for curve in case['curves']:
            low,high = curve['mean_squared_error_difference_bounds']
            if previous:
                assert low <= previous[0] and high >= previous[1]
            previous = low,high
            if curve['all_scored_reference_locations_inside_disk']:
                actual = case['reference_mean_squared_error_difference']
                assert low-1e-8 <= actual <= high+1e-8
