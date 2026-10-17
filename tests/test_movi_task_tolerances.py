import json
from pathlib import Path

import numpy as np
import pytest

from paper.analyze_movi_task_tolerances import compare, run


def test_tolerance_crossing_and_unavailable_denominator():
    result=compare([1,10,np.inf],[2,3,20])
    assert result['curves_cross']
    assert result['paired']==3 and result['unavailable_dage']==1
    assert result['dage_success_counts'][-1]==2
    assert result['da3_success_counts'][-1]==3
    assert compare([1,2],[2,3])['dage_weakly_dominates']
    assert compare([np.inf],[np.inf])['dage_weakly_dominates']
    with pytest.raises(ValueError):
        compare([np.nan],[1])


def test_saved_task_curves_replay():
    root=Path(__file__).resolve().parents[1]/'paper/results/movi-camera-pilot-v1'
    result=run(root)
    assert result==json.loads((root/'task-tolerances-v1.json').read_text())
    assert len(result['results'])==16
    for row in result['results']:
        assert all(0<=n<=row['paired'] for n in row['dage_success_counts']+row['da3_success_counts'])
        assert np.all(np.diff(row['dage_success_counts'])>=0)
        assert np.all(np.diff(row['da3_success_counts'])>=0)
