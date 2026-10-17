import json
from pathlib import Path

import numpy as np

from paper.probe_movi_error_regression import features, fit, predict, run


def test_ridge_training_and_prediction_do_not_read_prediction_labels():
    rows = [dict(sequence='s',eligible=True,error=float(i),features=[float(i)]) for i in range(3)]
    model = fit(rows,[0])
    row = dict(features=[1.5],error=0,sequence='unseen')
    before = predict(model,row)
    row['error'] = 1e9
    assert predict(model,row) == before
    constant = fit(rows,[])
    assert np.isclose(predict(constant,row),np.expm1(np.mean(np.log1p([0,1,2]))))
    assert len(features([12,12],[13,14],12,.5,None)) == 7


def test_saved_error_regression_replay():
    root=Path(__file__).resolve().parents[1]/'paper/results'
    result=run(root)
    assert result == json.loads((root/'movi-replication-shard01-v1/error-regression-v1.json').read_text())
    for method in result['results'].values():
        assert len(method['model']['training_scenes']) == 4
        assert all(len(rows)==768 for rows in method['records'].values())
        for rows in method['records'].values():
            assert all(r['predicted_error_pixels'] is None for r in rows if not r['eligible'])
