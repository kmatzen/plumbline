"""Freeze RGB-only camera-marker proposals before any reference-motion audit."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.run_movi_cameras import load_rgb
from paper.decode_movi_geometry_pilot import digest
from paper.tapvid_camera_markers import fit_marker, project
from plumbline.conventions import assert_valid_extrinsics

QUERIES = (12, 18, 23)


def track_rgb(video):
    """Fixed raster pixels, chained LK with a one-pixel backward check."""
    if video.shape != (24, 256, 256, 3) or video.dtype != np.uint8:
        raise ValueError('expected all 24 native RGB frames')
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
    seeds = np.array([(x, y) for y in range(12, 256, 32) for x in range(12, 256, 32)], np.float32)
    tracks = [dict(track=i, query_frame=0, query_xy=xy.tolist(), records=[])
              for i, xy in enumerate(seeds)]
    previous = seeds.copy()
    alive = np.ones(64, bool)
    for frame in range(1, 24):
        indices = np.flatnonzero(alive)
        observations = {}
        if len(indices):
            initial = previous[indices].reshape(-1, 1, 2)
            current, status, _ = cv2.calcOpticalFlowPyrLK(gray[frame-1], gray[frame], initial, None,
                                                       winSize=(21, 21), maxLevel=3)
            if current is not None:
                good = status.reshape(-1).astype(bool) & np.isfinite(current).all(axis=(1, 2))
                safe = current.copy()
                safe[~good] = initial[~good]
                back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray[frame], gray[frame-1], safe, None,
                                                                  winSize=(21, 21), maxLevel=3)
                if back is not None:
                    error = np.linalg.norm(back-initial, axis=(1, 2))
                    good &= backward_status.reshape(-1).astype(bool) & np.isfinite(error) & (error <= 1)
                    good &= ((current >= 0) & (current < 256)).all(axis=(1, 2))
                    for slot, xy, fb, accepted in zip(indices, current[:, 0], error, good, strict=True):
                        if accepted:
                            observations[int(slot)] = (xy, float(fb))
        alive[:] = False
        for slot, track in enumerate(tracks):
            value = observations.get(slot)
            if value is not None:
                previous[slot] = value[0]
                alive[slot] = True
            track['records'].append(dict(frame=frame, xy=None if value is None else value[0].tolist(),
                forward_backward_pixels=None if value is None else value[1]))
    return tracks


def marker_rows(track, poses, calibration):
    fit = fit_marker(track, poses, calibration, np.array([256, 256]))
    rows = []
    for frame in QUERIES:
        row = dict(frame=frame, status=fit['status'], candidate_xy=None)
        if fit['status'] == 'fitted':
            xy = project(np.array(fit['point_world']), poses[frame], calibration[frame], np.array([256, 256]))
            if xy is None:
                row['status'] = 'nonpositive_query_depth'
            elif not np.isfinite(xy).all():
                row['status'] = 'nonfinite_query_projection'
            else:
                row.update(candidate_xy=xy.tolist(), status='renderable' if np.all((np.rint(xy) >= 0) & (np.rint(xy) < 256)) else 'query_outside_image')
        rows.append(row)
    return fit, rows


def build(root, runs, sequence, output, expected_names=None):
    if output.exists():
        raise ValueError('preserve previous geometry')
    video, provenance = load_rgb(root, sequence, expected_names)
    tracks = track_rgb(video)
    records, fits, cameras = [], [], {}
    for method in ('dage', 'da3'):
        directory = runs/f'{sequence}-{method}'
        manifest_path = directory/'manifest.json'
        camera = json.loads(manifest_path.read_text())
        if camera['sequence'] != sequence or camera['method'] != method or camera['status'] not in ('complete', 'failed'):
            raise ValueError('camera identity or terminal status mismatch')
        cameras[method] = dict(manifest_sha256=digest(manifest_path), status=camera['status'])
        if camera['status'] == 'complete':
            if any(camera[key] != value for key, value in provenance.items()):
                raise ValueError('camera RGB provenance mismatch')
            path = directory/'prediction.npz'
            if digest(path) != camera['prediction_sha256']:
                raise ValueError('prediction hash mismatch')
            with np.load(path, allow_pickle=False) as arrays:
                poses, calibration = arrays['extrinsics'], arrays['intrinsics_normalized']
            if poses.shape != (24, 4, 4) or calibration.shape != (24, 3, 3) or not np.isfinite(calibration).all():
                raise ValueError('invalid camera arrays')
            assert_valid_extrinsics(poses)
            cameras[method]['prediction_sha256'] = digest(path)
        for track in tracks:
            if camera['status'] == 'complete':
                fit, rows = marker_rows(track, poses, calibration)
            else:
                fit = dict(status='camera_run_failed')
                rows = [dict(frame=f, status='camera_run_failed', candidate_xy=None) for f in QUERIES]
            fits.append(dict(method=method, track=track['track'], **fit))
            for row in rows:
                identity = hashlib.sha256(f"movi-marker-v1:{sequence}:{method}:{track['track']}:{row['frame']}".encode()).hexdigest()[:24]
                records.append(dict(id=identity, sequence=sequence, method=method, track=track['track'],
                    source_xy=track['query_xy'], source_frame=0, **row))
    result = dict(scope='RGB-only development marker proposals; not camera-quality or DNF labels',
        policy='64 fixed pixels x,y=12,44,...,236; all slots retained; chained RGB LK, FB<=1; fit frames 0..9 only; query frames 12,18,23; no lost-track replacement; predicted intrinsics',
        limitations='no texture or staticness certification; later RGB tracks are saved but never used to fit markers; all-frame camera inference is not causal',
        sequence=sequence, provenance=provenance, cameras=cameras, tracks=tracks, fits=fits,
        planned_presentations=384, counts=dict(Counter(r['status'] for r in records)), records=records,
        opencv_version=cv2.__version__, script_sha256=digest(Path(__file__)),
        dependencies_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('run_movi_cameras.py', 'tapvid_camera_markers.py', 'build_anchor_preview.py')})
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('sequence')
    parser.add_argument('output', type=Path)
    parser.add_argument('--sequences', nargs='+')
    args = parser.parse_args()
    print(build(args.root, args.runs, args.sequence, args.output, args.sequences)['counts'])
