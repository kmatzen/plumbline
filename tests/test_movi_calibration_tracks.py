import json
from pathlib import Path

import pytest
import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.run_movi_calibration_tracks import validate_shards
from paper.probe_movi_reverse_tracking import cycle_answer


def test_only_completed_calibration_shards_can_be_opened(tmp_path):
    plan=Path(__file__).resolve().parents[1]/'paper/results/movi-uncertainty-calibration-v1/protocol.json'
    (tmp_path/'batch.json').write_text(json.dumps(dict(protocol_sha256=digest(plan),attempts=[])))
    with pytest.raises(ValueError,match='calibration-only'):
        validate_shards(tmp_path,plan,[27])
    with pytest.raises(ValueError,match='completed acquisition'):
        validate_shards(tmp_path,plan,[2])
    decoded=tmp_path/'shard-00002/decoded-v1';decoded.mkdir(parents=True)
    path=decoded/'manifest.json'
    path.write_text(json.dumps(dict(status='complete',shard_index=2,expected_scenes=1,scenes=[dict(name='test')])))
    (tmp_path/'batch.json').write_text(json.dumps(dict(protocol_sha256=digest(plan),attempts=[
        dict(shard_index=2,status='complete',decoded_manifest_sha256=digest(path),scenes=['test'])])))
    assert validate_shards(tmp_path,plan,[2])[0][2]==['test']
    (tmp_path/'shard-00002/image-evidence-v1').mkdir()
    with pytest.raises(FileExistsError):
        validate_shards(tmp_path,plan,[2])


def test_first_completed_calibration_tracks_are_complete_and_replay():
    paper=Path(__file__).resolve().parents[1]/'paper'
    root=paper/'results/movi-uncertainty-calibration-v1'
    prior=json.loads((paper/'results/movi-temporal-verifier-v1/manifest.json').read_text())
    scene_names=set()
    for index in (2,3):
        folder=root/f'shard-{index:05d}'
        evidence=folder/'image-evidence-v1'
        saved=json.loads((evidence/'manifest.json').read_text())
        assert saved['status']=='complete'
        assert saved['checkpoint_sha256']==prior['checkpoint_sha256']
        assert saved['source_files_sha256']==prior['source_files_sha256']
        assert saved['script_sha256']==digest(paper/'run_movi_calibration_tracks.py')
        assert saved['protocol_sha256']==digest(root/'protocol.json')
        assert saved['uncertainty_model_sha256']==digest(root/'model.json')
        assert saved['decoded_manifest_sha256']==digest(folder/'decoded-v1/manifest.json')
        assert len(saved['records'])==saved['planned_observations']==768
        assert not scene_names&set(saved['sequences'])
        scene_names.update(saved['sequences'])
        for scene in saved['scenes']:
            assert digest(evidence/scene['forward_file'])==scene['forward_sha256']
            with np.load(evidence/scene['forward_file'],allow_pickle=False) as a:
                rows=[r for r in saved['records'] if r['sequence']==scene['sequence']]
                assert {(r['frame'],r['track']) for r in rows}=={(f,t) for f in (12,18,23) for t in range(64)}
                for r in rows:
                    assert r['source_xy']==[12+32*(r['track']%8),12+32*(r['track']//8)]
                    np.testing.assert_array_equal(r['predicted_xy'],a['xy'][r['frame'],r['track']])
                    assert r['predicted_visible']==bool(a['visibility'][r['frame'],r['track']])
                    if r['reverse_attempted']:
                        error=float(np.linalg.norm(np.array(r['reverse_xy'],np.float32)-np.array(r['source_xy'],np.float32)))
                        assert error==r['cycle_error_pixels']
                    else:
                        assert r['cycle_error_pixels'] is None and not r['reverse_visible']
                    assert r['cycle_predicted']==cycle_answer(r['predicted_visible'],r['reverse_visible'],r['cycle_error_pixels'])
    assert len(scene_names)==8
