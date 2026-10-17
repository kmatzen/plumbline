import json
from pathlib import Path

import numpy as np

from paper.build_movi_marker_geometry import marker_rows
from paper.decode_movi_geometry_pilot import digest


def test_prefix_only_temporal_proposals_replay():
    paper=Path(__file__).resolve().parents[1]/'paper'
    root=paper/'results/movi-temporal-markers-v1'
    original=paper/'results/movi-camera-pilot-v1'
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert manifest['script_sha256']==digest(paper/'build_movi_temporal_markers.py')
    count=0
    for scene in manifest['scenes']:
        path=root/scene['file'];assert digest(path)==scene['sha256']
        saved=json.loads(path.read_text());source=original/scene['file']
        assert saved['source_geometry_sha256']==digest(source)
        old=json.loads(source.read_text())
        assert {r['id'] for r in saved['records']}=={r['id'] for r in old['records']}
        for method in ('dage','da3'):
            with np.load(original/'runs'/f"{scene['sequence']}-{method}"/'prediction.npz') as a:
                for track in saved['tracks']:
                    assert [r['frame'] for r in track['records']]==list(range(1,10))
                    fit,rows=marker_rows(track,a['extrinsics'],a['intrinsics_normalized'])
                    for row in rows:
                        found=next(r for r in saved['records'] if r['method']==method and r['track']==track['track'] and r['frame']==row['frame'])
                        assert row['status']==found['status']
                        if row['candidate_xy'] is not None:np.testing.assert_allclose(row['candidate_xy'],found['candidate_xy'],atol=1e-7)
                        else:assert found['candidate_xy'] is None
                        count+=1
    assert count==1536
