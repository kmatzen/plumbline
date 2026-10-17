"""Reference-prefix ablation of marker placement; never evaluator inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.tapvid_camera_markers import fit_marker, project


def oracle_fit(xy, poses, k):
    track = dict(query_frame=0, query_xy=xy[0].tolist(),
        records=[dict(frame=i, xy=xy[i].tolist()) for i in range(1, 10)])
    return fit_marker(track, poses, k, np.array([256, 256]))


def run(root, background):
    manifest_path = background/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old_path = root/'background-marker-audit-v1.json'
    old = json.loads(old_path.read_text())
    assert old['background_manifest_sha256'] == digest(manifest_path)
    lookup = {r['id']:r for r in old['records']}
    rows, sources = [], {}
    for scene in manifest['scenes']:
        name = scene['name']
        path = background/scene['file']
        assert digest(path) == scene['sha256']
        geometry_path = root/f'marker-geometry-{name}-v1.json'
        assert digest(geometry_path) == old['sources'][name]['geometry_sha256']
        geometry = json.loads(geometry_path.read_text())
        sources[name] = dict(reference=digest(path), geometry=digest(geometry_path))
        with np.load(path, allow_pickle=False) as a:
            refs = a['projected_xy'].copy()
            eligible = a['in_image'] & ~a['depth_rejected'] & ~a['mask_rejected']
            slots = {int(slot):i for i,slot in enumerate(a['slots'])}
        for method in ('dage', 'da3'):
            prediction = root/'runs'/f'{name}-{method}'/'prediction.npz'
            assert digest(prediction) == geometry['cameras'][method]['prediction_sha256']
            with np.load(prediction, allow_pickle=False) as a:
                poses, k = a['extrinsics'], a['intrinsics_normalized']
            fits = {slot:oracle_fit(refs[index], poses, k) for slot,index in slots.items()}
            for original in geometry['records']:
                if original['method'] != method:
                    continue
                row = dict(id=original['id'], sequence=name, method=method, track=original['track'],
                    frame=original['frame'], status='foreground_source', rgb_error_pixels=None,
                    oracle_error_pixels=None, prefix_reference_eligible=False, query_reference_eligible=False)
                rows.append(row)
                index = slots.get(original['track'])
                if index is None:
                    continue
                fit = fits[original['track']]
                frame = original['frame']
                row.update(status=fit['status'], prefix_reference_eligible=bool(eligible[index,:10].all()),
                    query_reference_eligible=bool(eligible[index,frame]), oracle_fit=fit,
                    rgb_error_pixels=lookup[row['id']].get('candidate_to_background_pixels'))
                if fit['status'] == 'fitted':
                    xy = project(np.array(fit['point_world']), poses[frame], k[frame], np.array([256,256]))
                    if xy is None or not np.isfinite(xy).all():
                        row['status'] = 'undefined_query_projection'
                    else:
                        row.update(status='projected', oracle_error_pixels=float(np.linalg.norm(xy-refs[index,frame])))
    summaries = []
    for name in sources:
        for method in ('dage','da3'):
            planned = [r for r in rows if r['sequence']==name and r['method']==method]
            paired = [r for r in planned if r['prefix_reference_eligible'] and r['query_reference_eligible']
                and r['rgb_error_pixels'] is not None and r['oracle_error_pixels'] is not None]
            summaries.append(dict(sequence=name, method=method, planned=len(planned), paired=len(paired),
                excluded_from_paired=len(planned)-len(paired),
                median_rgb_error_pixels=float(np.median([r['rgb_error_pixels'] for r in paired])) if paired else None,
                median_oracle_error_pixels=float(np.median([r['oracle_error_pixels'] for r in paired])) if paired else None,
                median_paired_rgb_minus_oracle_pixels=float(np.median([r['rgb_error_pixels']-r['oracle_error_pixels'] for r in paired])) if paired else None))
    return dict(scope='reference-assisted prefix tracking ablation on development scenes; not reference-free metric or DNF',
        protocol='all 1536 slots retained; replace early RGB point coordinates with exact background projections on frames 0..9; unchanged predicted cameras, intrinsics and fitting helper; query 12/18/23; no later coordinates used in fitting; keep off-image errors',
        limitation='reference visibility remains diagnostic; per-method paired subsets differ, so summaries do not establish a method ranking; oracle fit still contains camera/intrinsics and fitting errors; differences are not an additive causal decomposition',
        script_sha256=digest(Path(__file__)), sources=sources, original_audit_sha256=digest(old_path),
        summaries=summaries, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','background','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    result=run(args.root,args.background)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summaries'],indent=2))
