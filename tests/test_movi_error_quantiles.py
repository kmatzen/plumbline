import json
from pathlib import Path

import numpy as np

from paper.probe_movi_error_quantiles import fit,run
from paper.probe_movi_error_regression import predict


def test_constant_quantile_matches_known_order_statistic():
    rows=[dict(sequence='s',eligible=True,error=float(i),features=[0.]) for i in range(11)]
    model=fit(rows,[],quantile=.9)
    assert np.isclose(predict(model,rows[0]),9.)
    rows[0]['error']=1000
    assert np.isclose(predict(model,rows[0]),9.)


def test_saved_quantile_probe_replays():
    root=Path(__file__).resolve().parents[1]/'paper/results/movi-replication-shard01-v1'
    result=run(root/'error-regression-v1.json')
    assert result==json.loads((root/'error-quantiles-v1.json').read_text())
    for method in result['results'].values():
        assert len(method['model']['training_scenes'])==4
        assert all(len(rows)==768 for rows in method['records'].values())
