import json
from pathlib import Path

import numpy as np

from paper.audit_movi_prefix_tracking import oracle_fit, run
from paper.tapvid_camera_markers import project


def test_exact_prefix_recovers_point_and_ignores_future_coordinates():
    poses = np.tile(np.eye(4),(24,1,1))
    poses[:,0,3] = np.linspace(0,1,24)
    k = np.tile([[1.,0,.5],[0,1.,.5],[0,0,1.]],(24,1,1))
    point = np.array([.2,.3,4.])
    xy = np.array([project(point,p,c,np.array([256,256])) for p,c in zip(poses,k)])
    fitted = oracle_fit(xy,poses,k)
    assert fitted['status'] == 'fitted'
    np.testing.assert_allclose(fitted['point_world'],point,atol=1e-9)
    xy[10:] = 999
    assert oracle_fit(xy,poses,k) == fitted


def test_saved_ablation_replays_and_retains_full_roster():
    paper = Path(__file__).resolve().parents[1]/'paper'
    root = paper/'results/movi-camera-pilot-v1'
    # Keep paths identical to the command that generated the portable report.
    result = run(root, paper/'results/movi-geometry-pilot-v1/background-audit-v1')
    saved = json.loads((root/'prefix-tracking-ablation-v1.json').read_text())
    assert result == saved
    assert len(result['records']) == len({r['id'] for r in result['records']}) == 1536
    assert sum(s['planned'] for s in result['summaries']) == 1536
    assert all(s['paired']+s['excluded_from_paired']==s['planned'] for s in result['summaries'])
