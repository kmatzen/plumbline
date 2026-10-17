"""Check whether visible annotated details lie inside supplied candidate crops."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def in_crop(reference, candidate):
    offset=np.asarray(reference,float)-np.rint(candidate)
    return bool(np.all((offset>=-80)&(offset<=80)))


def run(staged, corpus):
    staging=json.loads((staged/'manifest.json').read_text())
    manifest=json.loads((corpus/'manifest.json').read_text())
    assert manifest['source_manifest_sha256']==digest(staged/'manifest.json')
    rows=[];sources={}
    allowed=[r for r in manifest['records'] if r['partition'] in ('train','validation') and r['status']=='labeled']
    for name in sorted({r['sequence'] for r in allowed}):
        assert manifest['partitions'][name] in ('train','validation')
        entry=staging['sequences'][name]
        path=staged/entry['file']
        assert digest(path)==entry['file_sha256']
        sources[name]=digest(path)
        with np.load(path,allow_pickle=False) as a:
            points,occluded=a['points'],a['occluded']
        height,width=entry['video_shape'][1:3]
        for row in allowed:
            if row['sequence']!=name: continue
            visible=not bool(occluded[row['track'],row['frame']])
            assert visible==(row['label']!='occluded')
            reference=points[row['track'],row['frame']]*[width,height]
            rows.append(dict(id=row['id'],sequence=name,partition=row['partition'],label=row['label'],
                candidate_crop_contains_visible_reference=in_crop(reference,row['candidate_xy']) if visible else None,
                reference_visible=visible))
    summaries={}
    for partition in ('train','validation'):
        subset=[r for r in rows if r['partition']==partition]
        summaries[partition]=dict(total=len(subset),label_counts=dict(Counter(r['label'] for r in subset)),
            visible_far_outside_crop=sum(r['label']=='far' and not r['candidate_crop_contains_visible_reference'] for r in subset),
            visible_far_inside_crop=sum(r['label']=='far' and r['candidate_crop_contains_visible_reference'] for r in subset))
    return dict(scope='development evidence-support audit; not visual observability labels or camera evaluation',
        policy='only train/validation source files; candidate crop is rounded center plus/minus 80 pixels; do not inspect occluded point coordinates for containment; no calibration images or exports opened',
        implication='a visible detail outside the supplied crop cannot be located within that crop; its absence alone is not evidence of occlusion; geometric containment does not guarantee perceptual identifiability',
        script_sha256=digest(Path(__file__)),corpus_sha256=digest(corpus/'manifest.json'),source_sha256=sources,
        summaries=summaries,records=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('staged','corpus','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();result=run(args.staged,args.corpus)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summaries'],indent=2))
