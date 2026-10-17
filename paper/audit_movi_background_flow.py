"""Check every adjacent step of frozen background references against rendered flow."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def sample_flow(flow, xy):
    """Bilinear sampling in integer-centered pixel coordinates; no extrapolation."""
    height, width = flow.shape[:2]
    result = np.full((len(xy), 2), np.nan)
    valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] <= width-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= height-1)
    p = xy[valid]
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo+1, [width-1, height-1])
    w = p-lo
    result[valid] = ((1-w[:, 0, None])*(1-w[:, 1, None])*flow[lo[:, 1], lo[:, 0]]
        + w[:, 0, None]*(1-w[:, 1, None])*flow[lo[:, 1], hi[:, 0]]
        + (1-w[:, 0, None])*w[:, 1, None]*flow[hi[:, 1], lo[:, 0]]
        + w[:, 0, None]*w[:, 1, None]*flow[hi[:, 1], hi[:, 0]])
    return result


def compare(background, encoded, bounds):
    xy = background['projected_xy']
    frames = xy.shape[1]
    if encoded.shape[0] != frames or encoded.shape[-1] != 2:
        raise ValueError('flow/trajectory frame mismatch')
    flow = bounds[0]+encoded.astype(float)/65535*(bounds[1]-bounds[0])
    # Dataset stores row displacement first. Do not change source pixel origin.
    measured = np.stack([sample_flow(flow[t, :, :, ::-1], xy[:, t]) for t in range(frames-1)], axis=1)
    expected = xy[:, 1:]-xy[:, :-1]
    finite = np.isfinite(measured).all(axis=-1)
    supported = background['in_image'] & ~background['depth_rejected'] & ~background['mask_rejected']
    interior = background['matching_background_neighbors'] == 4
    strict = finite & supported[:, :-1] & supported[:, 1:] & interior[:, :-1] & interior[:, 1:]
    return dict(source_xy=background['source_xy'], projected_displacement_xy=expected,
                sampled_rendered_flow_xy=measured, flow_error_pixels=np.linalg.norm(expected-measured, axis=-1),
                sampled=finite, depth_mask_interior_pair=strict,
                source_depth_mask_not_rejected=supported[:, :-1], target_depth_mask_not_rejected=supported[:, 1:])


def quantiles(values):
    return np.quantile(values, [0, .5, .95, 1]).tolist() if len(values) else None


def run(source, background_root, output):
    from tfrecord import tfrecord_iterator, example_pb2
    manifest_path = background_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    shard = source/'movi_e-validation.tfrecord-00000-of-00064'
    if (manifest['status'] != 'complete' or digest(shard) != manifest['shard_sha256']
            or [s['name'] for s in manifest['scenes']] != ['3835', '5397', '4385', '4970']):
        raise ValueError('complete fixed background audit required')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='all-step background reference consistency; not independent visibility truth or camera DNF',
        policy='all 195 original background sources and 23 adjacent pairs; bilinear rendered flow sampled at frozen reference pixels; preserve unavailable and depth/mask rejected cases; summarize all finite samples and fixed interior/non-rejected pair subset separately',
        limitations='rendered flow and depth share a source; interpolation near motion/depth boundaries can disagree; consistency is not a human label or evaluator accuracy',
        source_manifest_sha256=digest(manifest_path), shard_sha256=digest(shard), script_sha256=digest(Path(__file__)), scenes=[])
    try:
        for index, payload in enumerate(tfrecord_iterator(str(shard))):
            example = example_pb2.Example(); example.ParseFromString(payload)
            f = example.features.feature
            name = f['metadata/video_name'].bytes_list.value[0].decode()
            entry = manifest['scenes'][index]
            if name != entry['name']:
                raise ValueError('scene roster changed')
            path = background_root/entry['file']
            if digest(path) != entry['sha256']:
                raise ValueError('background records changed')
            raw = np.asarray(f['forward_flow'].int64_list.value)
            if raw.size != 24*256*256*2 or np.any((raw < 0) | (raw > 65535)):
                raise ValueError('invalid flow array')
            encoded = raw.reshape(24, 256, 256, 2).astype(np.uint16)
            bounds = np.asarray(f['metadata/forward_flow_range'].float_list.value)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('invalid flow range')
            with np.load(path, allow_pickle=False) as background:
                records = compare(background, encoded, bounds)
            target = output/f'{index:02d}-flow-consistency.npz'
            np.savez_compressed(target, **records)
            result['scenes'].append(dict(name=name, file=target.name, sha256=digest(target),
                background_sha256=digest(path), planned_pairs=int(records['sampled'].size),
                sampled_pairs=int(records['sampled'].sum()), interior_nonrejected_pairs=int(records['depth_mask_interior_pair'].sum()),
                all_sampled_error_quantiles_pixels=quantiles(records['flow_error_pixels'][records['sampled']]),
                interior_nonrejected_error_quantiles_pixels=quantiles(records['flow_error_pixels'][records['depth_mask_interior_pair']])))
        if len(result['scenes']) != 4:
            raise ValueError('incomplete shard')
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'background', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.background, args.output)['scenes'], indent=2))
