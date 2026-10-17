import json
from pathlib import Path

import numpy as np

from paper.audit_movi_motion_track_error import residuals, run


def test_reference_changes_only_oracle_residual():
    f = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    before = residuals(f, [10, 20], [30, 20], [30, 20])
    after = residuals(f, [10, 20], [30, 20], [30, 30])
    assert before['predicted_sampson_pixels'] == after['predicted_sampson_pixels'] == 0
    assert before['oracle_sampson_pixels'] == 0
    assert np.isclose(after['oracle_sampson_pixels'], 10/np.sqrt(2))
    assert after['tracking_error_pixels'] == 10


def test_saved_intervention_replay_and_visibility_accounting():
    root = Path(__file__).resolve().parents[1]/'paper/results'
    verifier = root/'movi-temporal-verifier-v1'
    result = run(root/'movi-camera-pilot-v1', root/'movi-geometry-pilot-v1', verifier)
    assert result == json.loads((verifier/'motion-track-error-v1.json').read_text())
    assert result['results']['all']['planned'] == 768
    assert sum(result['results'][key]['planned'] for key in
               ('predicted_visible', 'predicted_invisible')) == 768
