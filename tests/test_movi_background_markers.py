import json
from pathlib import Path

from paper.audit_movi_background_markers import run


def test_archived_audit_replays_and_keeps_full_roster():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    result = run(root/'movi-camera-pilot-v1', root/'movi-geometry-pilot-v1/background-audit-v1')
    saved = json.loads((root/'movi-camera-pilot-v1/background-marker-audit-v1.json').read_text())
    assert result == saved
    assert len(result['records']) == len({r['id'] for r in result['records']}) == 1536
    assert result['counts'] == dict(background_source=1170, foreground_source=366)
    assert all(r['status'] != 'renderable' or 'candidate_to_background_pixels' in r
               for r in result['records'] if r['background_status'] == 'background_source')
