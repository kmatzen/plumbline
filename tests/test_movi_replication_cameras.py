import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest
from paper.run_movi_replication_cameras import run


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
