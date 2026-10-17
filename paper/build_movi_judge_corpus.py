"""Unlabeled MOVi panels and RGB baselines; no reference arrays are opened."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.build_tapvid_judge_corpus import PROMPT, render
from paper.decode_movi_geometry_pilot import digest
from paper.probe_tapvid_identity import correlation, patch
from paper.run_movi_cameras import load_rgb


def baseline(candidate, observation, source_gray, later_gray, source_xy):
    xy = observation['xy']
    ncc = None if xy is None else correlation(patch(source_gray, source_xy), patch(later_gray, xy))
    distance = None if xy is None else float(np.linalg.norm(np.asarray(candidate)-xy))
    accepted = (xy is not None and observation['forward_backward_pixels'] is not None
                and observation['forward_backward_pixels'] <= 1 and ncc is not None and ncc >= .8)
    answer = ('close' if distance <= 2 else 'far' if distance >= 8 else 'uncertain') if accepted else 'uncertain'
    return dict(rgb_baseline_answer=answer, rgb_track_ncc=ncc, distance_to_rgb_track=distance,
                rgb_track_accepted=accepted)


def build(decoded, geometry_root, output):
    if output.exists():
        raise ValueError('preserve previous corpus')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    records, sources = [], {}
    for name in ('3835', '5397', '4385', '4970'):
        video, provenance = load_rgb(decoded, name)
        geometry_path = geometry_root/f'marker-geometry-{name}-v1.json'
        geometry = json.loads(geometry_path.read_text())
        if geometry['provenance'] != provenance or geometry['sequence'] != name:
            raise ValueError('geometry/RGB provenance mismatch')
        expected = {(m, s, f) for m in ('dage', 'da3') for s in range(64) for f in (12, 18, 23)}
        if len(geometry['records']) != 384 or {(r['method'], r['track'], r['frame']) for r in geometry['records']} != expected:
            raise ValueError('full fixed roster required')
        sources[name] = digest(geometry_path)
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
        for original in geometry['records']:
            row = dict(original, rgb_baseline_answer='uncertain', rgb_track_ncc=None,
                       distance_to_rgb_track=None, rgb_track_accepted=False)
            if row['status'] == 'renderable':
                image = render(video[0], video[row['frame']], row['source_xy'], row['candidate_xy'])
                path = output/'images'/f"{row['id']}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('panel write failed')
                row.update(image=str(path.relative_to(output)), image_sha256=digest(path))
                track = geometry['tracks'][row['track']]
                if track['track'] != row['track'] or track['query_xy'] != row['source_xy']:
                    raise ValueError('RGB track/source mismatch')
                observation = next(r for r in track['records'] if r['frame'] == row['frame'])
                row.update(baseline(row['candidate_xy'], observation, gray[0], gray[row['frame']], row['source_xy']))
            records.append(row)
    export = output/'inference.jsonl'
    export.write_text(''.join(json.dumps(dict(id=r['id'], image=r['image'], prompt=PROMPT))+'\n'
                             for r in records if r['status'] == 'renderable'))
    result = dict(application='movi_marker_development',
        scope='unlabeled development inference; no ground truth in model inputs; no camera-DNF labels',
        policy='all 1536 fixed proposals; all renderable panels; original prompt, renderer and base/adapted judge; no threshold tuning, training or calibration access',
        rgb_baseline_policy='saved later LK observation, FB<=1 and original-source 17x17 patch NCC>=0.8; distance<=2 close, >=8 far, otherwise uncertain; no occlusion prediction',
        timing='comparison added after reference geometry and motion audits; development only, not a preregistered or held-out test',
        prompt=PROMPT, planned_presentations=1536, records=records,
        counts=dict(Counter(r['status'] for r in records)),
        baseline_counts=dict(Counter(r['rgb_baseline_answer'] for r in records)),
        source_geometry_sha256=sources, export_sha256=digest(export), script_sha256=digest(Path(__file__)),
        source_manifest_sha256=digest(decoded/'manifest.json'),
        dependencies_sha256={n:digest(Path(__file__).with_name(n)) for n in
                             ('build_tapvid_judge_corpus.py', 'probe_tapvid_identity.py', 'run_movi_cameras.py')})
    (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded', 'geometry', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = build(args.decoded, args.geometry, args.output)
    print(result['counts'], result['baseline_counts'])
