"""Frozen reverse-prefix tracking check on all original MOVi grid queries."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def eligible(row, width, height):
    x, y = row['predicted_xy']
    return bool(row['predicted_visible'] and np.isfinite([x, y]).all()
                and 0 <= x <= width-1 and 0 <= y <= height-1)


def cycle_answer(forward_visible, reverse_visible, error):
    if not forward_visible or not reverse_visible or error is None:
        return 'unresolved'
    return 'consistent' if error <= 1 else 'inconsistent'


def run(decoded, camera, forward_path, repo, checkpoint, output):
    import torch
    from paper.run_movi_cameras import load_rgb

    forward = json.loads(forward_path.read_text())
    if forward['status'] != 'complete' or digest(checkpoint) != forward['checkpoint_sha256']:
        raise ValueError('forward run/checkpoint mismatch')
    for name, sha in forward['source_files_sha256'].items():
        if digest(repo/name) != sha:
            raise ValueError('tracker source changed')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running',
        scope='development forward/reverse consistency baseline, not correctness certification or camera DNF',
        protocol='all 768 unique fixed point/frame slots; original forward predictions frozen; reverse RGB prefix from query frame to frame zero; forward-visible in-image endpoints queried without clipping; upstream CoTracker3 defaults unchanged; reverse visibility required; fixed 1px cycle threshold; no reference labels enter inference',
        limitation='same tracker in both directions has correlated errors; forward inference used all 24 frames but reverse uses prefix only; consistency is not correctness',
        script_sha256=digest(Path(__file__)), forward_sha256=digest(forward_path),
        checkpoint_sha256=digest(checkpoint), source_files_sha256=forward['source_files_sha256'],
        scenes=[], records=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60).eval().cuda()
    try:
        for name in ('3835', '5397', '4385', '4970'):
            geometry_path = camera/f'marker-geometry-{name}-v1.json'
            geometry = json.loads(geometry_path.read_text())
            saved = next(s for s in forward['scenes'] if s['sequence'] == name)
            if digest(geometry_path) != saved['geometry_sha256']:
                raise ValueError('query geometry changed')
            video, provenance = load_rgb(decoded, name)
            if provenance != saved['provenance']:
                raise ValueError('RGB changed')
            sources = {t['track']: t['query_xy'] for t in geometry['tracks']}
            for frame in (12, 18, 23):
                pair = {method: sorted([r for r in forward['records']
                    if r['sequence'] == name and r['frame'] == frame and r['method'] == method],
                    key=lambda r: r['track']) for method in ('dage', 'da3')}
                rows = pair['dage']
                if ([r['track'] for r in rows] != list(range(64))
                    or [(r['predicted_xy'], r['predicted_visible']) for r in rows]
                    != [(r['predicted_xy'], r['predicted_visible']) for r in pair['da3']]):
                    raise ValueError('full identical source query roster required')
                active = [r for r in rows if eligible(r, video.shape[2], video.shape[1])]
                returned = {}
                if active:
                    batch = torch.from_numpy(video[:frame+1][::-1].copy()).permute(0,3,1,2)[None].float().cuda()
                    queries = torch.tensor([[[0, *r['predicted_xy']] for r in active]], dtype=torch.float32, device='cuda')
                    with torch.inference_mode():
                        xy, visibility = model(batch, queries=queries, backward_tracking=False)
                    endpoints = xy[0, -1].cpu().numpy()
                    visible = visibility[0, -1].cpu().numpy()
                    if not np.isfinite(endpoints).all() or visible.dtype != bool:
                        raise ValueError('invalid reverse predictions')
                    returned = {r['track']: (endpoints[i], bool(visible[i])) for i, r in enumerate(active)}
                    del batch, queries, xy, visibility
                    torch.cuda.empty_cache()
                for row in rows:
                    point = row['track']
                    endpoint, visible = returned.get(point, (None, False))
                    error = None if endpoint is None else float(np.linalg.norm(endpoint-np.asarray(sources[point])))
                    result['records'].append(dict(sequence=name, frame=frame, track=point,
                        source_xy=sources[point], forward_xy=row['predicted_xy'],
                        forward_visible=row['predicted_visible'], reverse_attempted=point in returned,
                        reverse_xy=None if endpoint is None else endpoint.tolist(), reverse_visible=visible,
                        cycle_error_pixels=error, predicted=cycle_answer(row['predicted_visible'], visible, error)))
                save()
                print(name, frame, len(active), len(result['records']), flush=True)
            result['scenes'].append(saved)
        if len(result['records']) != 768:
            raise ValueError('incomplete roster')
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded', 'camera', 'forward', 'repo', 'checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.decoded, args.camera, args.forward, args.repo, args.checkpoint, args.output)
