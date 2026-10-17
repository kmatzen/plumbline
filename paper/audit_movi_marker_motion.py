"""Join frozen RGB proposals to separate development motion diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def join(records, temporal, visibility):
    # Temporal raster centers have +.5; RGB/LK pixels use integer centers.
    index = {tuple(xy-.5): i for i, xy in enumerate(temporal['source_xy'])}
    rows = []
    for record in records:
        row = dict(record)
        rows.append(row)
        point = index.get(tuple(record['source_xy']))
        if point is None:
            row['reference_status'] = 'no_foreground_reference'
            continue
        frame = record['frame']
        row.update(reference_status='foreground_reference',
                   visibility_evidence='supported' if visibility['supported'][point, frame] else
                       'rejected' if visibility['rejected'][point, frame] else 'unresolved',
                   world_displacement=float(temporal['world_displacement'][point, frame]),
                   counterfactual_static_error_pixels=float(temporal['counterfactual_static_error_pixels'][point, frame]))
        if record['candidate_xy'] is not None:
            reference_xy = temporal['projected_xy'][point, frame]-.5
            row['candidate_to_moving_surface_pixels'] = float(np.linalg.norm(np.asarray(record['candidate_xy'])-reference_xy))
    return rows


def run(geometry_root, audit_root):
    temporal_path = audit_root/'temporal-audit-v1/manifest.json'
    visibility_path = audit_root/'visibility-audit-v1/manifest.json'
    temporal_manifest = json.loads(temporal_path.read_text())
    visibility_manifest = json.loads(visibility_path.read_text())
    if visibility_manifest['source_sha256']['temporal'] != digest(temporal_path):
        raise ValueError('visibility/temporal provenance mismatch')
    rows, sources = [], {}
    for scene in temporal_manifest['scenes']:
        name = scene['name']
        path = geometry_root/f'marker-geometry-{name}-v1.json'
        geometry = json.loads(path.read_text())
        if geometry['provenance']['input_manifest_sha256'] != temporal_manifest['source_manifest_sha256']:
            raise ValueError('geometry/reference inputs mismatch')
        if len(geometry['records']) != 384 or len({r['id'] for r in geometry['records']}) != 384:
            raise ValueError('all fixed proposals required')
        vscene = next(s for s in visibility_manifest['scenes'] if s['name'] == name)
        tpath = temporal_path.parent/scene['records_file']
        vpath = visibility_path.parent/vscene['records_file']
        if digest(tpath) != scene['records_sha256'] or digest(vpath) != vscene['records_sha256']:
            raise ValueError('reference arrays changed')
        with np.load(tpath, allow_pickle=False) as temporal, np.load(vpath, allow_pickle=False) as visibility:
            rows.extend(join(geometry['records'], temporal, visibility))
        sources[name] = dict(geometry=digest(path), temporal=digest(tpath), visibility=digest(vpath))
    return dict(scope='development motion audit only; visibility cues are not independently verified labels; not camera DNF',
                policy='join all frozen proposals; retain background, unavailable and uncertain visibility; continuous motion, no fitted staticness threshold',
                script_sha256=digest(Path(__file__)), source_sha256=sources,
                temporal_manifest_sha256=digest(temporal_path), visibility_manifest_sha256=digest(visibility_path),
                counts=dict(Counter(r['reference_status'] for r in rows)),
                visibility_counts=dict(Counter(r.get('visibility_evidence', 'no_reference') for r in rows)), records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'audit', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.geometry, args.audit)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['counts'], result['visibility_counts'])
