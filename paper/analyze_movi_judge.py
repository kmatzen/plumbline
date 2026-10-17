"""Diagnostic agreement with MOVi reference cues, not certified camera accuracy."""

import argparse
import json
from collections import Counter
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest
from paper.evaluate_tapvid_lora import parse_label

GEOMETRY_FIELDS = ('id', 'sequence', 'method', 'track', 'frame', 'source_frame', 'source_xy', 'status', 'candidate_xy')


def index(records):
    result = {r['id']:r for r in records}
    if len(result) != len(records):
        raise ValueError('duplicate IDs')
    return result


def references(corpus, background, motion):
    original, bg, fg = [index(d['records']) for d in (corpus, background, motion)]
    if set(original) != set(bg) or set(original) != set(fg) or len(original) != 1536:
        raise ValueError('full paired reference roster required')
    rows = []
    for identity, record in original.items():
        if any(record[k] != other[identity][k] for other in (bg, fg) for k in GEOMETRY_FIELDS):
            raise ValueError('geometry changed during reference audit')
        b, f = bg[identity], fg[identity]
        is_background = b['background_status'] == 'background_source'
        if is_background != (f['reference_status'] == 'no_foreground_reference'):
            raise ValueError('background/foreground references disagree')
        row = dict(record, region='background' if is_background else 'foreground',
                   diagnostic_label=None, scoring_status='geometry_unavailable')
        if record['status'] == 'renderable':
            usable = b['reference_depth_mask_not_rejected'] if is_background else f['visibility_evidence'] == 'supported'
            row['scoring_status'] = 'reference_visibility_unresolved_or_rejected'
            if usable:
                error = b['candidate_to_background_pixels'] if is_background else f['candidate_to_moving_surface_pixels']
                row['reference_distance_pixels'] = error
                label = 'close' if error <= 2 else 'far' if error >= 8 else None
                row.update(diagnostic_label=label, scoring_status='diagnostic_labeled' if label else 'borderline')
        rows.append(row)
    return rows


def agreement(rows, answers):
    labeled = [r for r in rows if r['diagnostic_label'] is not None]
    confusion = {label:dict(Counter(answers[r['id']] for r in labeled if r['diagnostic_label'] == label))
                 for label in ('close', 'far')}
    answered = [r for r in labeled if answers[r['id']] not in ('uncertain', 'invalid')]
    correct = sum(answers[r['id']] == r['diagnostic_label'] for r in labeled)
    predicted_close = [r for r in labeled if answers[r['id']] == 'close']
    true_close = sum(r['diagnostic_label'] == 'close' for r in predicted_close)
    return dict(planned=len(rows), diagnostic_labeled=len(labeled),
        counts=dict(Counter(r['scoring_status'] for r in rows)),
        all_planned_answers=dict(Counter(answers[r['id']] for r in rows)), confusion=confusion,
        answered=len(answered), correct=correct,
        answer_coverage_on_diagnostic_labels=len(answered)/len(labeled) if labeled else None,
        agreement_on_all_diagnostic_labels=correct/len(labeled) if labeled else None,
        conditional_agreement=correct/len(answered) if answered else None,
        close_precision_on_diagnostic_labels=true_close/len(predicted_close) if predicted_close else None)


def judge_answers(corpus, evaluation):
    if evaluation['status'] != 'complete' or set(evaluation['results']) != {'base', 'adapted'}:
        raise ValueError('wait for both complete arms')
    expected = {r['id'] for r in corpus['records'] if r['status'] == 'renderable'}
    paired = {arm:index(records) for arm, records in evaluation['results'].items()}
    for arm, records in paired.items():
        if set(records) != expected:
            raise ValueError('complete identical renderable roster required')
        for row in records.values():
            if row['target_for_scoring_only'] is not None or row['parsed'] != parse_label(row['raw_response']):
                raise ValueError('target leakage or parser mismatch')
    for identity in expected:
        first, second = paired['base'][identity], paired['adapted'][identity]
        if any(first[k] != second[k] for k in ('processed_pixel_sha256', 'image_tokens', 'sequence')):
            raise ValueError('paired model input mismatch')
    return {arm:{r['id']:records[r['id']]['parsed'] if r['id'] in expected else 'unavailable'
                 for r in corpus['records']} for arm, records in paired.items()}


def run(root, judge=None):
    corpus_path = root/'judge-corpus-v1/manifest.json'
    bg_path, fg_path = root/'background-marker-audit-v1.json', root/'marker-motion-audit-v1.json'
    corpus, bg, fg = [json.loads(p.read_text()) for p in (corpus_path, bg_path, fg_path)]
    for name, sha in corpus['source_geometry_sha256'].items():
        if (digest(root/f'marker-geometry-{name}-v1.json') != sha
                or bg['sources'][name]['geometry_sha256'] != sha
                or fg['source_sha256'][name]['geometry'] != sha):
            raise ValueError('reference/corpus geometry hashes disagree')
    rows = references(corpus, bg, fg)
    arms = dict(rgb_track={r['id']:r['rgb_baseline_answer'] for r in rows})
    sources = dict(corpus=digest(corpus_path), background=digest(bg_path), motion=digest(fg_path),
                   analysis=digest(Path(__file__)))
    if judge is not None:
        evaluation = json.loads(judge.read_text())
        if (evaluation['evaluation_corpus_manifest_sha256'] != digest(corpus_path)
                or evaluation['evaluation_export_sha256'] != corpus['export_sha256']):
            raise ValueError('judge corpus mismatch')
        arms.update(judge_answers(corpus, evaluation))
        sources['judge'] = digest(judge)
    baseline_answered = {r['id'] for r in rows if arms['rgb_track'][r['id']] in ('close', 'far')}
    strata = {'all':rows,
        'background':[r for r in rows if r['region'] == 'background'],
        'foreground':[r for r in rows if r['region'] == 'foreground'],
        'rgb_answered':[r for r in rows if r['id'] in baseline_answered],
        'rgb_abstained':[r for r in rows if r['id'] not in baseline_answered]}
    strata.update({f'scene_{name}':[r for r in rows if r['sequence'] == name] for name in corpus['source_geometry_sha256']})
    return dict(scope='development diagnostic agreement; not independent visibility truth, camera accuracy, DNF, or held-out validation',
        policy='all 1536 proposals retained; 2/8 pixel thresholds unchanged; only reference depth/mask non-rejected backgrounds and normal-supported foregrounds receive diagnostic close/far labels; all other references remain unresolved or borderline; no occluded labels inferred',
        limitations='reference cues can be wrong; repeated frames and both methods are dependent; no sample-level significance claim; missing cases are never counted as camera failures',
        source_sha256=sources, planned_presentations=len(rows),
        results={arm:{name:agreement(subset, answers) for name, subset in strata.items()} for arm, answers in arms.items()},
        diagnostic_records=[dict(id=r['id'], sequence=r['sequence'], method=r['method'], region=r['region'],
                                 scoring_status=r['scoring_status'], diagnostic_label=r['diagnostic_label']) for r in rows])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--judge', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.judge)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({arm:r['all'] for arm,r in result['results'].items()}, indent=2))
