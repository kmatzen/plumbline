import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def test_temporal_baseline_archive_and_fixed_decisions():
    paper=Path(__file__).resolve().parents[1]/'paper'
    saved=json.loads((paper/'results/tapvid-cotracker-verifier-v1/manifest.json').read_text())
    corpus=json.loads((paper/'results/tapvid-judge-corpus-v1/manifest.json').read_text())
    assert saved['status']=='complete'
    assert saved['script_sha256']==digest(paper/'probe_tapvid_cotracker_verifier.py')
    assert saved['corpus_sha256']==digest(paper/'results/tapvid-judge-corpus-v1/manifest.json')
    expected={r['id']:r for r in corpus['records'] if r['partition'] in ('train','validation') and r['status']=='labeled'}
    assert len(saved['records'])==len(expected)==316
    assert {r['id'] for r in saved['records']}==set(expected)
    assert all(w['status']=='complete' and 0<w['last']-w['first']<=60 for w in saved['windows'])
    for row in saved['records']:
        assert row['target_for_scoring_only']==expected[row['id']]['label']
        d=row['candidate_distance_pixels']
        answer='occluded' if not row['predicted_visible'] else 'close' if d<=2 else 'far' if d>=8 else 'uncertain'
        assert row['predicted']==answer
    for split,n in [('train',167),('validation',99)]:
        assert sum(r['predicted']==r['target_for_scoring_only'] for r in saved['records'] if r['partition']==split)==n
