"""Apply frozen tracking, motion and reverse checks to the replication roster."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_crossfit_support import crossfit
from paper.probe_movi_reverse_tracking import cycle_answer, eligible
from paper.run_movi_replication_cameras import roster


def queries(geometry):
    expected = [[float(x), float(y)] for y in range(12, 256, 32) for x in range(12, 256, 32)]
    tracks = geometry['tracks']
    if ([r['track'] for r in tracks] != list(range(64))
            or [r['query_xy'] for r in tracks] != expected):
        raise ValueError('fixed source grid changed')
    return np.array([[0, *xy] for xy in expected], np.float32)


def run(decoded, camera, protocol, repo, checkpoint, prior_path, output):
    import torch
    from paper.run_movi_cameras import load_rgb

    names = roster(decoded, protocol)
    prior = json.loads(prior_path.read_text())
    if prior['status'] != 'complete' or digest(checkpoint) != prior['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')
    for filename, sha in prior['source_files_sha256'].items():
        if digest(repo/filename) != sha:
            raise ValueError('tracker source changed')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='prospective fixed-check replication; no reference geometry used in inference',
        protocol_sha256=digest(protocol), input_manifest_sha256=digest(decoded/'manifest.json'),
        prior_sha256=digest(prior_path), checkpoint_sha256=digest(checkpoint),
        source_files_sha256=prior['source_files_sha256'], script_sha256=digest(Path(__file__)),
        helper_sha256={n: digest(Path(__file__).with_name(n)) for n in
            ('probe_movi_crossfit_support.py', 'probe_movi_rigid_support.py', 'probe_movi_reverse_tracking.py', 'run_movi_cameras.py')},
        policy='same 64 source pixels, offline full 24-frame forward inference; fixed four-fold crossfit matrices at frames 12,18,23; reverse prefixes from visible in-image endpoints; 1px cycle and Sampson thresholds; no candidate camera or reference geometry enters tracker or motion fit',
        sequences=names, scenes=[], records=[], fits=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60).eval().cuda()
    try:
        for name in names:
            path = camera/f'marker-geometry-{name}-v1.json'
            geometry = json.loads(path.read_text())
            video, provenance = load_rgb(decoded, name, names)
            if provenance != geometry['provenance']:
                raise ValueError('RGB provenance mismatch')
            seeds = queries(geometry)
            batch = torch.from_numpy(video.copy()).permute(0,3,1,2)[None].float().cuda()
            with torch.inference_mode():
                xy, visibility = model(batch, queries=torch.from_numpy(seeds)[None].cuda(), backward_tracking=False)
            xy = xy[0].cpu().numpy()
            visibility = visibility[0].cpu().numpy()
            if xy.shape != (24,64,2) or visibility.shape != (24,64) or not np.isfinite(xy).all() or visibility.dtype != bool:
                raise ValueError('invalid forward output')
            del batch
            arrays = output/f'{name}-forward.npz'
            np.savez_compressed(arrays, xy=xy, visibility=visibility)
            for frame in (12,18,23):
                motion = crossfit(seeds[:,1:], xy[frame], visibility[frame])
                result['fits'].append(dict(sequence=name, frame=frame, **motion))
                points = [dict(track=i, predicted_xy=xy[frame,i].tolist(), predicted_visible=bool(visibility[frame,i])) for i in range(64)]
                active = [r for r in points if eligible(r,256,256)]
                returned = {}
                if active:
                    batch = torch.from_numpy(video[:frame+1][::-1].copy()).permute(0,3,1,2)[None].float().cuda()
                    reverse_queries = torch.tensor([[[0,*r['predicted_xy']] for r in active]], dtype=torch.float32, device='cuda')
                    with torch.inference_mode():
                        back, seen = model(batch, queries=reverse_queries, backward_tracking=False)
                    endpoints, visible = back[0,-1].cpu().numpy(), seen[0,-1].cpu().numpy()
                    if not np.isfinite(endpoints).all() or visible.dtype != bool:
                        raise ValueError('invalid reverse output')
                    returned = {r['track']: (endpoints[i], bool(visible[i])) for i,r in enumerate(active)}
                    del batch, reverse_queries, back, seen
                for point in points:
                    i = point['track']
                    endpoint, visible = returned.get(i, (None,False))
                    error = None if endpoint is None else float(np.linalg.norm(endpoint-seeds[i,1:]))
                    result['records'].append(dict(sequence=name, frame=frame, **point,
                        source_xy=seeds[i,1:].tolist(), reverse_attempted=i in returned,
                        reverse_xy=None if endpoint is None else endpoint.tolist(), reverse_visible=visible,
                        cycle_error_pixels=error, cycle_predicted=cycle_answer(point['predicted_visible'],visible,error),
                        motion_predicted=motion['answers'][i], motion_sampson_pixels=motion['distances'][i]))
                save()
                torch.cuda.empty_cache()
            result['scenes'].append(dict(sequence=name, geometry_sha256=digest(path), provenance=provenance,
                                        forward_file=arrays.name, forward_sha256=digest(arrays)))
            save()
            print(name, len(result['records']), flush=True)
        if len(result['records']) != len(names)*192:
            raise ValueError('incomplete roster')
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded', 'camera', 'protocol', 'repo', 'checkpoint', 'prior', 'output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    run(args.decoded,args.camera,args.protocol,args.repo,args.checkpoint,args.prior,args.output)
