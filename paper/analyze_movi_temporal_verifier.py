"""Replay identical diagnostic subsets for the frozen temporal verifier."""

import argparse
import json
from pathlib import Path

from paper.analyze_movi_judge import agreement, run as baseline_analysis
from paper.decode_movi_geometry_pilot import digest


def run(root, predictions):
    saved=json.loads(predictions.read_text())
    path=root/'judge-corpus-v1/manifest.json'
    corpus=json.loads(path.read_text())
    assert saved['status']=='complete' and saved['corpus_sha256']==digest(path)
    expected={r['id']:r for r in corpus['records']}
    assert len(saved['records'])==len(expected)==1536
    assert {r['id'] for r in saved['records']}==set(expected)
    answers={}
    for row in saved['records']:
        original=expected[row['id']]
        assert all(row[k]==original[k] for k in ('sequence','method','frame','track'))
        distance=row['candidate_distance_pixels']
        answer='unavailable' if original['status']!='renderable' else 'occluded' if not row['predicted_visible'] else 'close' if distance<=2 else 'far' if distance>=8 else 'uncertain'
        assert answer==row['predicted']
        answers[row['id']]=answer
    reference=baseline_analysis(root)
    rows=reference['diagnostic_records']
    strata={'all':rows,'rgb_abstained':[r for r in rows if expected[r['id']]['rgb_baseline_answer']=='uncertain']}
    strata.update({f'scene_{s}':[r for r in rows if r['sequence']==s] for s in ('3835','5397','4385','4970')})
    return dict(scope=reference['scope'],limitations=reference['limitations'],policy=reference['policy'],
        source_sha256=dict(predictions=digest(predictions),analysis=digest(Path(__file__)),reference=reference['source_sha256']),
        results={k:agreement(v,answers) for k,v in strata.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','predictions','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();result=run(args.root,args.predictions)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result['results'].items() if k in ('all','rgb_abstained')},indent=2))
