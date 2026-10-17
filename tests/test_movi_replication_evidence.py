import json
from pathlib import Path

import numpy as np
import pytest

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_crossfit_support import crossfit
from paper.probe_movi_reverse_tracking import cycle_answer
from paper.run_movi_replication_evidence import queries


def test_replication_grid_does_not_depend_on_camera_results():
    geometry = dict(tracks=[dict(track=i, query_xy=[float(x),float(y)])
        for i,(x,y) in enumerate((x,y) for y in range(12,256,32) for x in range(12,256,32))])
    before = queries(geometry)
    geometry['records'] = [{'status':'failed', 'candidate_xy':None}]
    assert (queries(geometry) == before).all()
    assert before.shape == (64,3)
    geometry['tracks'][0]['query_xy'][0] += 1
    with pytest.raises(ValueError, match='grid changed'):
        queries(geometry)


def test_retained_replication_evidence_replays_frozen_checks():
    paper = Path(__file__).resolve().parents[1]/'paper'
    root = paper/'results/movi-replication-shard01-v1'
    folder = root/'evidence-v1'
    saved = json.loads((folder/'manifest.json').read_text())
    assert saved['status'] == 'complete'
    assert saved['script_sha256'] == digest(paper/'run_movi_replication_evidence.py')
    assert saved['protocol_sha256'] == digest(root/'protocol.json')
    assert saved['input_manifest_sha256'] == digest(root/'decoded-manifest.json')
    assert len(saved['records']) == len({(r['sequence'],r['frame'],r['track']) for r in saved['records']}) == 768
    for scene in saved['scenes']:
        name = scene['sequence']
        geometry_path = root/f'marker-geometry-{name}-v1.json'
        assert scene['geometry_sha256'] == digest(geometry_path)
        seeds = queries(json.loads(geometry_path.read_text()))[:,1:]
        assert scene['forward_sha256'] == digest(folder/scene['forward_file'])
        with np.load(folder/scene['forward_file'], allow_pickle=False) as a:
            for frame in (12,18,23):
                replay = crossfit(seeds,a['xy'][frame],a['visibility'][frame])
                rows = sorted([r for r in saved['records'] if r['sequence'] == name and r['frame'] == frame],key=lambda r:r['track'])
                assert [r['motion_predicted'] for r in rows] == replay['answers']
                for r in rows:
                    i = r['track']
                    np.testing.assert_array_equal(r['predicted_xy'],a['xy'][frame,i])
                    assert r['predicted_visible'] == bool(a['visibility'][frame,i])
                    error = r['cycle_error_pixels']
                    if r['reverse_attempted']:
                        assert np.isclose(error,np.linalg.norm(np.asarray(r['reverse_xy'],dtype=np.float32)-seeds[i]))
                    else:
                        assert error is None and not r['reverse_visible']
                    assert r['cycle_predicted'] == cycle_answer(r['predicted_visible'],r['reverse_visible'],error)
