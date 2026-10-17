import json
from pathlib import Path

import pytest

from paper.stage_movi_calibration import validate_protocol


def test_model_is_exact_frozen_export_and_evaluation_is_separate():
    root=Path(__file__).resolve().parents[1]/'paper/results'
    folder=root/'movi-uncertainty-calibration-v1'
    model=json.loads((folder/'model.json').read_text())
    original=json.loads((root/'movi-replication-shard01-v1/error-quantiles-v1.json').read_text())
    assert model['model']==original['results']['cycle_quantile']['model']
    plan=json.loads((folder/'protocol.json').read_text())
    assert validate_protocol(plan)==list(range(2,27))
    info=json.loads((root/'movi-geometry-pilot-v1/dataset_info.json').read_text())
    counts=next(s for s in info['splits'] if s['name']=='validation')['shardLengths']
    assert sum(int(counts[i]) for i in plan['calibration_shards'])==97
    assert sum(int(counts[i]) for i in plan['reserved_evaluation_shards'])==40
    plan['calibration_shards'].append(27)
    with pytest.raises(ValueError):
        validate_protocol(plan)
