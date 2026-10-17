import json
from pathlib import Path

from paper.analyze_movi_reverse_tracking import run
from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_reverse_tracking import cycle_answer, eligible


def test_reverse_query_eligibility_does_not_clip_or_accept_hidden_points():
    assert eligible(dict(predicted_xy=[0, 255], predicted_visible=True), 256, 256)
    assert not eligible(dict(predicted_xy=[0, 256], predicted_visible=True), 256, 256)
    assert not eligible(dict(predicted_xy=[0, 10], predicted_visible=False), 256, 256)


def test_cycle_threshold_and_unresolved_evidence():
    assert cycle_answer(True, True, 1.) == 'consistent'
    assert cycle_answer(True, True, 1.001) == 'inconsistent'
    assert cycle_answer(True, False, 0.) == 'unresolved'
    assert cycle_answer(False, True, 0.) == 'unresolved'
    assert cycle_answer(True, True, None) == 'unresolved'


def test_completed_reverse_experiment_and_exact_diagnostic_replay():
    root = Path(__file__).resolve().parents[1]
    folder = root/'paper/results/movi-reverse-tracking-v1'
    saved = json.loads((folder/'manifest.json').read_text())
    assert saved['status'] == 'complete'
    assert saved['script_sha256'] == digest(root/'paper/probe_movi_reverse_tracking.py')
    assert len(saved['scenes']) == 4
    result = run(folder/'manifest.json', root/'paper/results/movi-temporal-verifier-v1')
    assert result == json.loads((folder/'diagnostic.json').read_text())
    assert sum(result['results']['all']['cycle_answers'].values()) == 768
    assert all(r['epipolar_and_cycle'] == 'unresolved' or
               (r['epipolar_only'] == 'retained' and r['cycle_answer'] == 'consistent')
               for r in result['records'])
