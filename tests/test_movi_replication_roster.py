import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def test_new_shard_is_complete_disjoint_and_matches_frozen_selection():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    folder = root/'movi-replication-shard01-v1'
    plan = json.loads((folder/'protocol.json').read_text())
    staged = json.loads((folder/'manifest.json').read_text())
    decoded = json.loads((folder/'decoded-manifest.json').read_text())
    old = json.loads((root/'movi-geometry-pilot-v1/decoded-manifest.json').read_text())
    assert staged['status'] == decoded['status'] == 'complete'
    assert plan['shard_index'] == staged['shard_index'] == decoded['shard_index'] == 1
    assert plan['split'] == 'validation'
    assert decoded['source_manifest_sha256'] == digest(folder/'manifest.json')
    shard = next(r for r in staged['files'] if 'tfrecord' in r['file'])
    assert shard['file'] == 'movi_e-validation.tfrecord-00001-of-00064'
    assert shard['sha256'] == decoded['shard_sha256']
    assert len(decoded['scenes']) == decoded['expected_scenes'] == 4
    for field in ('name', 'record_sha256', 'decoded_rgb_sha256'):
        new_values = {s[field] for s in decoded['scenes']}
        assert len(new_values) == 4
        assert not new_values & {s[field] for s in old['scenes']}
