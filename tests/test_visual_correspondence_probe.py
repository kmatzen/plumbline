import json
from pathlib import Path

import pytest

from paper.decode_movi_geometry_pilot import digest
from paper.train_visual_correspondence_probe import load_split


def test_loader_restricts_partitions_and_checks_targets(tmp_path):
    image=tmp_path/'sample.png';image.write_bytes(b'fixture')
    row=dict(id='a',image='sample.png',target='close')
    source=tmp_path/'train.jsonl';source.write_text(json.dumps(row)+'\n')
    manifest=dict(exports_sha256={'train':digest(source)},records=[dict(
        id='a',image='sample.png',image_sha256=digest(image),partition='train',status='labeled',label='close')])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert load_split(tmp_path,'train')==[row]
    with pytest.raises(ValueError,match='only development'):
        load_split(tmp_path,'calibration')
    row['target']='far';source.write_text(json.dumps(row)+'\n')
    manifest['exports_sha256']['train']=digest(source)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='provenance'):
        load_split(tmp_path,'train')


def test_completed_visual_probe_lineage_and_counts():
    paper=Path(__file__).resolve().parents[1]/'paper'
    saved=json.loads((paper/'results/visual-correspondence-probe-v1/manifest.json').read_text())
    assert saved['status']=='complete'
    assert saved['script_sha256']==digest(paper/'train_visual_correspondence_probe.py')
    assert saved['corpus_sha256']==digest(paper/'results/tapvid-judge-corpus-v1/manifest.json')
    for arm,correct in [('frozen',60),('adapted',53)]:
        rows=saved['arms'][arm]['validation']
        assert len(rows)==len({r['id'] for r in rows})==128
        assert not set(saved['train_ids']) & {r['id'] for r in rows}
        assert sum(r['target']==r['predicted'] for r in rows)==correct
        assert len(saved['arms'][arm]['epochs'])==20
    assert [r['id'] for r in saved['arms']['frozen']['validation']]==[r['id'] for r in saved['arms']['adapted']['validation']]
