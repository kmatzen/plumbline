import numpy as np
import json
from pathlib import Path

from paper.probe_movi_rigid_support import estimate,sampson_distance,run


def test_epipolar_distance_and_missing_support():
    f=np.array([[0,0,0],[0,0,-1],[0,1,0]],float)
    first=np.array([[1,2],[3,4]],float);second=first+np.array([9,0])
    np.testing.assert_allclose(sampson_distance(f,first,second),0)
    second[:,1]+=2
    np.testing.assert_allclose(sampson_distance(f,first,second),np.sqrt(2))
    assert estimate(first,second,[True,False])['answers']==['unresolved','unresolved']


def test_camera_independent_rigid_support_replay():
    paper=Path(__file__).resolve().parents[1]/'paper'
    root=paper/'results/movi-temporal-verifier-v1'
    result=run(paper/'results/movi-camera-pilot-v1',root/'manifest.json')
    assert result==json.loads((root/'rigid-support-v1.json').read_text())
    assert len(result['records'])==768
    assert len({(r['sequence'],r['frame'],r['track']) for r in result['records']})==768
    assert result['confusion']['moving_foreground']==dict(incompatible=8,unresolved=5)
