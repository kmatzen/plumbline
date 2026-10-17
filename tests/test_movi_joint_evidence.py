import json
from pathlib import Path

from paper.audit_movi_joint_evidence import decide, run


def test_joint_gate_abstains_without_turning_missing_evidence_into_failure():
    for placement in ('occluded', 'unavailable', 'uncertain'):
        for support in ('compatible', 'incompatible', 'unresolved'):
            assert decide(placement, support, False) == 'unresolved'
            assert decide(placement, support, True) == 'unresolved'
    for placement in ('close', 'far'):
        assert decide(placement, 'compatible', True) == placement
        assert decide(placement, 'incompatible', True) == 'unresolved'
        assert decide(placement, 'unresolved', True) == 'unresolved'
        assert decide(placement, 'incompatible', False) == placement


def test_full_roster_and_scene_method_accounting():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    temporal = root/'movi-temporal-verifier-v1'
    result = run(root/'movi-camera-pilot-v1', temporal/'manifest.json',
                 temporal/'crossfit-support-v1.json')
    assert result == json.loads((temporal/'joint-evidence-v1.json').read_text())
    assert len(result['records']) == 1536
    assert len({r['id'] for r in result['records']}) == 1536
    for arm, strata in result['results'].items():
        assert sum(strata['all']['answers'].values()) == 1536
        assert sum(s['planned'] for k, s in strata.items() if k != 'all') == 1536
    assert all(r['crossfit_gated'] in (r['ungated'], 'unresolved')
               for r in result['records'])
