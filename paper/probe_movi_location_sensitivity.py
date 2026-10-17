"""Shared-location sensitivity on fixed replication placement comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from plumbline.paired_location_bounds import squared_difference_bounds


def run(root):
    ep, sp = root/'evidence-v1/manifest.json', root/'score-v1.json'
    evidence, score = [json.loads(p.read_text()) for p in (ep,sp)]
    if evidence['status'] != 'complete' or score['source_sha256']['evidence'] != digest(ep):
        raise ValueError('frozen inference/scoring mismatch')
    ref = {(r['sequence'],r['frame'],r['track']):r for r in score['points']}
    placements = {(r['sequence'],r['frame'],r['track'],r['method']):r for r in score['placements']}
    rows = []
    for scene in evidence['sequences']:
        path = root/f'marker-geometry-{scene}-v1.json'
        if digest(path) != score['geometry_sha256'][scene]:
            raise ValueError('candidate geometry changed')
        geometry = json.loads(path.read_text())
        candidates = {(r['track'],r['frame'],r['method']):r for r in geometry['records']}
        eligible = []
        for r in evidence['records']:
            if r['sequence'] != scene or not r['predicted_visible']:
                continue
            a,b = [candidates[r['track'],r['frame'],m] for m in ('dage','da3')]
            if a['status'] == b['status'] == 'renderable':
                eligible.append(dict(a=a['candidate_xy'],b=b['candidate_xy'],q=r['predicted_xy'],
                    reference=ref[scene,r['frame'],r['track']],frame=r['frame'],track=r['track']))
        for name, subset in (('all_forward_visible_pairs',eligible),
                             ('supported_background_subset',[r for r in eligible if r['reference']['reference'] == 'static_background'])):
            a,b,q = [np.array([r[k] for r in subset]) for k in ('a','b','q')]
            result = dict(sequence=scene,subset=name,planned=192,observations=len(subset),
                          selected_slots=[[r['frame'],r['track']] for r in subset],curves=[])
            if subset:
                max_error = (max(r['reference']['tracking_error_pixels'] for r in subset)
                             if name == 'supported_background_subset' else None)
                result['reference_max_tracking_error_pixels'] = max_error
                result['reference_mean_squared_error_difference'] = (float(np.mean([
                    placements[scene,r['frame'],r['track'],'dage']['reference_error_pixels']**2
                    -placements[scene,r['frame'],r['track'],'da3']['reference_error_pixels']**2
                    for r in subset])) if max_error is not None else None)
                denom = float(2*np.linalg.norm(a-b,axis=1).mean())
                center = float(np.mean(np.sum((a-q)**2-(b-q)**2,axis=1)))
                result['uniform_radius_to_zero_margin_pixels'] = abs(center)/denom if denom else None
                for radius in (0,.5,1,2,4,8):
                    bound = squared_difference_bounds(a,b,q,radius)
                    # Do not repeat a constant per-point radius array in the report.
                    bound['radius_pixels'] = radius
                    bound['all_scored_reference_locations_inside_disk'] = max_error <= radius if max_error is not None else None
                    result['curves'].append(bound)
            rows.append(result)
    return dict(scope='post-replication conditional sensitivity diagnostic, not independent validation, novel bound, or DNF',
                policy='fixed forward-visible common-renderable pairs, no motion/cycle filtering; radii 0,.5,1,2,4,8 pixels all reported; squared-error difference differs from prior mean Euclidean error and is not substituted for it',
                limitations='disks are hypothetical, not calibrated; references only check containment on supported background subset; missing or undefined tasks outside subset are not bounded; no full-plan ordering or human usability claim',
                source_sha256=dict(evidence=digest(ep),score=digest(sp)),
                script_sha256=digest(Path(__file__)),
                kernel_sha256=digest(Path(__file__).parents[1]/'src/plumbline/paired_location_bounds.py'),cases=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args = parser.parse_args(); result = run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in result['cases']:
        print(r['sequence'],r['subset'],r['observations'],r.get('uniform_radius_to_zero_margin_pixels'),r.get('reference_max_tracking_error_pixels'))
