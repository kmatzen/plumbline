import copy
import json
from pathlib import Path

import pytest

from paper.analyze_movi_judge import agreement, judge_answers, references, run


def test_abstention_and_unresolved_are_not_correct_answers():
    rows = [dict(id='a', diagnostic_label='close', scoring_status='diagnostic_labeled'),
            dict(id='b', diagnostic_label='far', scoring_status='diagnostic_labeled'),
            dict(id='c', diagnostic_label=None, scoring_status='reference_visibility_unresolved_or_rejected')]
    result = agreement(rows, dict(a='close', b='uncertain', c='close'))
    assert result['planned'] == 3 and result['diagnostic_labeled'] == 2
    assert result['conditional_agreement'] == 1
    assert result['agreement_on_all_diagnostic_labels'] == .5
    assert result['answer_coverage_on_diagnostic_labels'] == .5
    assert result['per_label_recall_including_abstentions'] == dict(close=1., far=0.)
    assert result['balanced_agreement_including_abstentions'] == .5
    assert agreement([], {})['conditional_agreement'] is None
    assert agreement(rows[:1], dict(a='close'))['balanced_agreement_including_abstentions'] is None


def test_model_pairing_leakage_and_partial_runs_rejected():
    corpus = dict(records=[dict(id='a', status='renderable'), dict(id='b', status='missing_prefix_observations')])
    record = dict(id='a', target_for_scoring_only=None, parsed='close', raw_response='close',
                  processed_pixel_sha256='pixels', image_tokens=10, sequence='3835')
    evaluation = dict(status='complete', results=dict(base=[record], adapted=[copy.deepcopy(record)]))
    assert judge_answers(corpus, evaluation)['base']['b'] == 'unavailable'
    evaluation['results']['adapted'][0]['processed_pixel_sha256'] = 'changed'
    with pytest.raises(ValueError, match='input mismatch'):
        judge_answers(corpus, evaluation)
    evaluation['status'] = 'evaluating'
    with pytest.raises(ValueError, match='complete'):
        judge_answers(corpus, evaluation)


def test_reference_roster_and_baseline_replay():
    root = Path(__file__).resolve().parents[1]/'paper/results/movi-camera-pilot-v1'
    result = run(root)
    saved = json.loads((root/'rgb-baseline-diagnostic-v2.json').read_text())
    assert result == saved
    all_rows = result['results']['rgb_track']['all']
    assert sum(all_rows['counts'].values()) == all_rows['planned'] == 1536
    assert all_rows['diagnostic_labeled'] == 389
    assert all_rows['correct'] == 244 and all_rows['answered'] == 247
    for arm, correct in (('constant_close', 352), ('constant_far', 37), ('constant_uncertain', 0)):
        assert result['results'][arm]['all']['correct'] == correct
    assert result['results']['constant_close']['all']['balanced_agreement_including_abstentions'] == .5
    corpus = json.loads((root/'judge-corpus-v1/manifest.json').read_text())
    background = json.loads((root/'background-marker-audit-v1.json').read_text())
    motion = json.loads((root/'marker-motion-audit-v1.json').read_text())
    background['records'][0]['candidate_xy'] = [100, 100]
    with pytest.raises(ValueError, match='geometry changed'):
        references(corpus, background, motion)
