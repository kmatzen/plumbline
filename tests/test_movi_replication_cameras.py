import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.run_movi_replication_cameras import run
from plumbline.conventions import assert_valid_extrinsics


def test_replication_runs_full_roster_after_process_failure(tmp_path, monkeypatch):
    root = tmp_path/'decoded'
    root.mkdir()
    names = ['7097', '4852', '2373', '4866']
    (root/'manifest.json').write_text(json.dumps(dict(status='complete', shard_index=1,
        expected_scenes=4, scenes=[dict(name=n) for n in names])))
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(status='frozen_before_shard_acquisition', split='validation',
        shard_index=1, camera_methods=['dage', 'da3'])))
    prior = tmp_path/'prior'
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'weights').write_bytes(b'fixed test weights')
    for method in ('dage', 'da3'):
        folder = prior/f'3835-{method}'
        folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(status='complete', method=method,
            checkpoint=str(checkpoint), checkpoint_files={'weights': digest(checkpoint/'weights')},
            settings={'fixed': True}, source_files={'fixed.py': 'unchanged'})))
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return 1
        output = Path(command[7])
        output.mkdir()
        (output/'manifest.json').write_text(json.dumps(dict(status='complete',
            settings={'fixed': True}, checkpoint_files={'weights': digest(checkpoint/'weights')},
            source_files={'fixed.py': 'unchanged'})))
        return 0
    monkeypatch.setattr('paper.run_movi_replication_cameras.subprocess.call', invoke)
    output = tmp_path/'runs'
    run(root, protocol, prior, output)
    saved = json.loads((output/'batch.json').read_text())
    assert saved['status'] == 'finished_all_attempts'
    assert len(calls) == len(saved['attempts']) == 8
    assert saved['attempts'][0]['status'] == 'process_failed'
    assert all(r['matches_prior_settings'] and r['matches_prior_checkpoint'] and
               r['matches_prior_model_sources'] for r in saved['attempts'][1:])
    assert all(c[-4:] == names for c in calls)


def test_retained_replication_cameras_match_inputs_and_frozen_settings():
    folder = Path(__file__).resolve().parents[1]/'paper/results/movi-replication-shard01-v1'
    batch = json.loads((folder/'runs/batch.json').read_text())
    assert batch['status'] == 'finished_all_attempts'
    assert batch['protocol_sha256'] == digest(folder/'protocol.json')
    assert batch['input_manifest_sha256'] == digest(folder/'decoded-manifest.json')
    assert len(batch['attempts']) == 8
    assert {(r['sequence'],r['method']) for r in batch['attempts']} == {
        (n,m) for n in batch['sequences'] for m in ('dage','da3')}
    for row in batch['attempts']:
        directory = folder/'runs'/row['output']
        manifest_path = directory/'manifest.json'
        assert digest(manifest_path) == row['manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete'
        assert manifest['input_manifest_sha256'] == batch['input_manifest_sha256']
        assert manifest['settings'] == batch['cameras'][row['method']]['settings']
        assert manifest['checkpoint_files'] == batch['cameras'][row['method']]['checkpoint_files']
        assert row['matches_prior_model_sources']
        prior = json.loads((folder.parent/'movi-camera-pilot-v1/runs'/f"3835-{row['method']}"/'manifest.json').read_text())
        assert manifest['source_files'] == prior['source_files']
        assert digest(directory/'prediction.npz') == manifest['prediction_sha256']
        with np.load(directory/'prediction.npz', allow_pickle=False) as arrays:
            assert arrays['extrinsics'].shape == (24,4,4)
            assert arrays['intrinsics_normalized'].shape == (24,3,3)
            assert np.isfinite(arrays['intrinsics_normalized']).all()
            assert_valid_extrinsics(arrays['extrinsics'])


def test_replication_marker_geometry_preserves_every_slot():
    folder = Path(__file__).resolve().parents[1]/'paper/results/movi-replication-shard01-v1'
    batch = json.loads((folder/'runs/batch.json').read_text())
    count = 0
    for name in batch['sequences']:
        geometry = json.loads((folder/f'marker-geometry-{name}-v1.json').read_text())
        assert len(geometry['records']) == 384
        assert len({r['id'] for r in geometry['records']}) == 384
        assert sum(geometry['counts'].values()) == 384
        assert {(r['method'],r['frame'],r['track']) for r in geometry['records']} == {
            (m,f,t) for m in ('dage','da3') for f in (12,18,23) for t in range(64)}
        for method in ('dage','da3'):
            assert geometry['cameras'][method]['prediction_sha256'] == digest(folder/'runs'/f'{name}-{method}'/'prediction.npz')
        count += geometry['counts'].get('renderable',0)
    assert count == 691
