import json
from pathlib import Path

import pytest

from paper.audit_movi_oracle_motion_control import run, unique_points


def test_reference_deduplication_rejects_method_dependent_motion():
    row = dict(sequence='s', frame=12, track=0, method='dage',
               counterfactual_static_error_pixels=10.)
    with pytest.raises(ValueError, match='exactly two'):
        unique_points([row])
    with pytest.raises(ValueError, match='reference mismatch'):
        unique_points([row, dict(row, method='da3', counterfactual_static_error_pixels=0.)])


def test_saved_control_replay_and_all_attempt_accounting():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    folder = root/'movi-temporal-verifier-v1'
    result = run(root/'movi-camera-pilot-v1', folder/'crossfit-support-v1.json')
    assert result == json.loads((folder/'oracle-motion-control-v1.json').read_text())
    assert len(result['records']) == 768
    for counts in result['results']['all']['answers'].values():
        assert sum(counts.values()) == 768
    for r in result['records']:
        if r['reference'] == 'unresolved':
            assert r['ungated'] == r['crossfit_gated'] == 'unresolved'
        if r['reference'] == 'static_background':
            assert r['ungated'] == 'close'
