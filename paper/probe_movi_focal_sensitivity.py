"""RGB-only focal sensitivity of early-fit marker forecasts, not DNF."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.tapvid_camera_markers import fit_marker, project

FACTORS = (.5, .8, 1., 1.25, 2.)
QUERIES = (12, 18, 23)


def probe(track, poses, calibration):
    rows = []
    for factor in FACTORS:
        k = calibration.astype(float).copy()
        k[:, 0, 0] *= factor
        k[:, 1, 1] *= factor
        fit = fit_marker(track, poses, k, np.array([256, 256]))
        forecasts = []
        for frame in QUERIES:
            xy = None if fit['status'] != 'fitted' else project(
                np.asarray(fit['point_world']), poses[frame], k[frame], np.array([256, 256]))
            forecasts.append(None if xy is None or not np.isfinite(xy).all() else xy.tolist())
        rows.append(dict(focal_factor=factor, fit=fit, forecasts=forecasts))
    base = rows[2]
    for row in rows:
        row['both_prefix_max_error_at_most_2px'] = all(
            r['fit']['status'] == 'fitted' and r['fit']['prefix_max_pixel_error'] <= 2
            for r in (base, row))
        row['forecast_difference_pixels'] = [
            None if a is None or b is None else float(np.linalg.norm(np.asarray(a)-b))
            for a, b in zip(base['forecasts'], row['forecasts'], strict=True)]
    return rows


def run(root):
    records, sources = [], {}
    for scene in ('3835', '5397', '4385', '4970'):
        geometry_path = root/f'marker-geometry-{scene}-v1.json'
        geometry = json.loads(geometry_path.read_text())
        sources[str(geometry_path)] = digest(geometry_path)
        if geometry['sequence'] != scene or [t['track'] for t in geometry['tracks']] != list(range(64)):
            raise ValueError('fixed scene and full track roster required')
        for method in ('dage', 'da3'):
            directory = root/'runs'/f'{scene}-{method}'
            manifest_path, prediction_path = directory/'manifest.json', directory/'prediction.npz'
            manifest = json.loads(manifest_path.read_text())
            if (digest(manifest_path) != geometry['cameras'][method]['manifest_sha256']
                    or digest(prediction_path) != geometry['cameras'][method]['prediction_sha256']
                    or manifest['status'] != 'complete'):
                raise ValueError('camera provenance mismatch')
            sources[str(prediction_path)] = digest(prediction_path)
            with np.load(prediction_path, allow_pickle=False) as arrays:
                poses, k = arrays['extrinsics'], arrays['intrinsics_normalized']
            for track in geometry['tracks']:
                records.append(dict(sequence=scene, method=method, track=track['track'],
                                    cases=probe(track, poses, k)))
    summaries = []
    for factor in FACTORS:
        cases = [next(c for c in r['cases'] if c['focal_factor'] == factor) for r in records]
        compatible = [c for c in cases if c['both_prefix_max_error_at_most_2px']]
        differences = [v for c in compatible for v in c['forecast_difference_pixels'] if v is not None]
        summaries.append(dict(focal_factor=factor, planned_tracks=len(cases),
            both_prefix_fits_at_most_2px=len(compatible), finite_query_pairs=len(differences),
            unavailable_query_pairs=3*len(compatible)-len(differences),
            query_pairs_disagreeing_at_least_8px=sum(v >= 8 for v in differences),
            maximum_forecast_difference_pixels=max(differences) if differences else None))
    return dict(scope='development focal sensitivity; not optimized camera alternatives, calibrated uncertainty, staticness, accuracy, or DNF',
        protocol='all 512 scene/method/track slots; fixed focal factors; poses unchanged; refit each point on frames 0..9 only; compare forecasts at 12/18/23; off-image projections retained; no reference arrays or later RGB coordinates read by fitting',
        interpretation='two small-residual fits with different forecasts are constructive ambiguity witnesses conditional on these fallible tracks; absence of a witness does not establish unique geometry; tracks can move or be wrong; factors are probes, not a calibrated prior',
        script_sha256=digest(Path(__file__)),
        helper_sha256={n:digest(Path(__file__).with_name(n)) for n in ('tapvid_camera_markers.py', 'build_anchor_preview.py')},
        source_sha256=sources, summaries=summaries, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summaries'], indent=2))
