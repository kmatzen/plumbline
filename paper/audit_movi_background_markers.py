"""Compare frozen camera proposals with separately reconstructed background tracks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def run(geometry_root, background_root):
    path = background_root/'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'complete' or [s['name'] for s in manifest['scenes']] != ['3835', '5397', '4385', '4970']:
        raise ValueError('complete fixed background audit required')
    rows, sources = [], {}
    for scene in manifest['scenes']:
        name = scene['name']
        geometry_path = geometry_root/f'marker-geometry-{name}-v1.json'
        geometry = json.loads(geometry_path.read_text())
        reference_path = background_root/scene['file']
        if (digest(reference_path) != scene['sha256'] or geometry['sequence'] != name
                or geometry['provenance']['input_manifest_sha256'] != manifest['decoded_manifest_sha256']
                or len(geometry['records']) != 384):
            raise ValueError('geometry/reference provenance mismatch')
        sources[name] = dict(geometry_sha256=digest(geometry_path), reference_sha256=digest(reference_path))
        with np.load(reference_path, allow_pickle=False) as a:
            lookup = {int(slot): i for i, slot in enumerate(a['slots'])}
            for record in geometry['records']:
                row = dict(record)
                rows.append(row)
                index = lookup.get(record['track'])
                if index is None:
                    row['background_status'] = 'foreground_source'
                    continue
                if not np.array_equal(a['source_xy'][index], record['source_xy']):
                    raise ValueError('source pixel changed')
                frame = record['frame']
                row.update(background_status='background_source',
                    reference_xy=a['projected_xy'][index, frame].tolist(),
                    reference_inside=bool(a['in_image'][index, frame]),
                    reference_depth_rejected=bool(a['depth_rejected'][index, frame]),
                    reference_mask_rejected=bool(a['mask_rejected'][index, frame]))
                row['reference_depth_mask_not_rejected'] = (row['reference_inside'] and
                    not row['reference_depth_rejected'] and not row['reference_mask_rejected'])
                if row['candidate_xy'] is not None:
                    row['candidate_to_background_pixels'] = float(np.linalg.norm(np.asarray(row['candidate_xy'])-row['reference_xy']))
                observation = next(r for r in geometry['tracks'][record['track']]['records'] if r['frame'] == frame)
                row['later_rgb_track_xy'] = observation['xy']
                if observation['xy'] is not None:
                    row['rgb_track_to_background_pixels'] = float(np.linalg.norm(np.asarray(observation['xy'])-row['reference_xy']))
                    if row['candidate_xy'] is not None:
                        row['candidate_to_rgb_track_pixels'] = float(np.linalg.norm(np.asarray(observation['xy'])-row['candidate_xy']))
    paired = {}
    for name in sources:
        methods = {m:{(r['track'], r['frame']):r for r in rows if r['sequence'] == name and r['method'] == m
            and r['status'] == 'renderable' and r.get('reference_depth_mask_not_rejected')}
                   for m in ('dage', 'da3')}
        common = sorted(set(methods['dage']) & set(methods['da3']))
        paired[name] = dict(common_proposals=len(common),
            median_candidate_to_background_pixels={m:float(np.median([methods[m][k]['candidate_to_background_pixels'] for k in common])) if common else None
                                                  for m in methods})
    return dict(scope='development background marker diagnostics, not independently validated visibility, DNF, or evaluator accuracy',
                policy='all 1536 original proposals retained; paired summaries use common renderable backgrounds not rejected by reference depth/masks; no RGB-track filtering or outcome-based replacement',
                limitations='marker error includes prefix tracking and intrinsics errors, not only camera error; summaries exclude unavailable proposals and must be read with coverage',
                background_manifest_sha256=digest(path), script_sha256=digest(Path(__file__)), sources=sources,
                counts=dict(Counter(r['background_status'] for r in rows)), paired_summaries=paired, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'background', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.geometry, args.background)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['paired_summaries'], indent=2))
