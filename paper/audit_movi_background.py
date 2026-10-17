"""Background reference tracks and rendered-flow consistency, separate from RGB inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.decode_movi_geometry_pilot import digest
from paper.audit_movi_camera_metadata import project_centers


def unproject(xy, radial, position, wxyz, focal_ratio, size):
    uv = (np.asarray(xy, float)+.5)/np.asarray(size)
    rays = np.column_stack([(uv[:, 0]-.5)/focal_ratio,
                            - (uv[:, 1]-.5)/focal_ratio, -np.ones(len(uv))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rotation = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()
    return (rays*np.asarray(radial)[:, None]) @ rotation.T + position


def background_arrays(a, encoded_flow, flow_range):
    masks = a['segmentations']
    frames, height, width = masks.shape
    xy = np.array([(x, y) for y in range(12, height, 32) for x in range(12, width, 32)])
    keep = masks[0, xy[:, 1], xy[:, 0]] == 0
    slots = np.flatnonzero(keep)
    xy = xy[keep]
    low, high = a['metadata_depth_range'].astype(float)
    depth = low+a['depth'].astype(float)/65535*(high-low)
    f = float(a['camera_focal_length'][0]/a['camera_sensor_width'][0])
    world = unproject(xy, depth[0, xy[:, 1], xy[:, 0]], a['camera_positions'][0],
                      a['camera_quaternions'][0], f, [width, height])
    world_frames = np.broadcast_to(world[:, None], (len(world), frames, 3))
    raster = project_centers(world_frames, a['camera_positions'], a['camera_quaternions'], f)*[width, height]
    rotation = Rotation.from_quat(a['camera_quaternions'][:, [1, 2, 3, 0]]).as_matrix()
    local = np.einsum('tji,ntj->nti', rotation, world_frames-a['camera_positions'][None])
    inside = (local[..., 2] < 0) & (raster[..., 0] >= .5) & (raster[..., 0] <= width-.5) & (raster[..., 1] >= .5) & (raster[..., 1] <= height-.5)
    radial = np.linalg.norm(world_frames-a['camera_positions'][None], axis=-1)
    maximum = np.full(radial.shape, np.nan)
    matches = np.zeros(radial.shape, int)
    for frame in range(frames):
        indices = np.flatnonzero(inside[:, frame])
        q = np.floor(raster[indices, frame]-.5).astype(int)
        neighbors, segments = [], []
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            x = np.clip(q[:, 0]+dx, 0, width-1)
            y = np.clip(q[:, 1]+dy, 0, height-1)
            neighbors.append(depth[frame, y, x])
            segments.append(masks[frame, y, x] == 0)
        maximum[indices, frame] = np.max(neighbors, axis=0)
        matches[indices, frame] = np.sum(segments, axis=0)
    # Rendered flow is (delta_row, delta_column), not (x,y).
    flow = flow_range[0]+encoded_flow.astype(float)/65535*(flow_range[1]-flow_range[0])
    expected = raster[:, 1]-raster[:, 0]
    measured = flow[xy[:, 1], xy[:, 0]][:, ::-1]
    return dict(slots=slots, source_xy=xy, world_point=world, projected_xy=raster-.5,
                radial_depth=radial, in_image=inside, matching_background_neighbors=matches,
                depth_rejected=inside & (maximum < .99*radial),
                mask_rejected=inside & (matches == 0), neighbor_depth_max=maximum,
                source_roundtrip_error_pixels=np.linalg.norm(raster[:, 0]-.5-xy, axis=1),
                rendered_flow_xy=measured, projected_first_displacement_xy=expected,
                flow_error_pixels=np.linalg.norm(expected-measured, axis=1))


def run(source, decoded, output):
    from tfrecord import tfrecord_iterator, example_pb2
    manifest_path = decoded/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    shard_index = manifest.get('shard_index', 0)
    if not isinstance(shard_index, int) or not 0 <= shard_index < 64:
        raise ValueError('invalid validation shard index')
    shard = source/f'movi_e-validation.tfrecord-{shard_index:05d}-of-00064'
    if manifest['status'] != 'complete' or digest(shard) != manifest['shard_sha256']:
        raise ValueError('fixed shard mismatch')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='background reference consistency only, not independently validated visibility or DNF',
        policy='fixed RGB-harness grid; background at frame 0; static world points from radial depth; all 24 frames; compare first displacement with rendered forward flow; retain every background source',
        limitations='depth and flow share the renderer; first-step agreement does not validate long-range visibility; depth/mask non-rejection is not certified visibility',
        sources=['https://raw.githubusercontent.com/google-research/kubric/main/challenges/point_tracking/dataset.py',
                 'https://raw.githubusercontent.com/google-research/kubric/main/challenges/movi/movi_e.py'],
        decoded_manifest_sha256=digest(manifest_path), shard_sha256=digest(shard),
        script_sha256=digest(Path(__file__)), scenes=[])
    try:
        for index, payload in enumerate(tfrecord_iterator(str(shard))):
            example = example_pb2.Example(); example.ParseFromString(payload)
            features = example.features.feature
            name = features['metadata/video_name'].bytes_list.value[0].decode()
            entry = manifest['scenes'][index]
            if name != entry['name']:
                raise ValueError('scene roster changed')
            path = decoded/entry['reference_file']
            if digest(path) != entry['reference_sha256']:
                raise ValueError('reference changed')
            raw = np.asarray(features['forward_flow'].int64_list.value)
            if raw.size != 24*256*256*2 or np.any((raw < 0) | (raw > 65535)):
                raise ValueError('invalid flow')
            flow = raw.reshape(24, 256, 256, 2)[0].astype(np.uint16)
            bounds = np.asarray(features['metadata/forward_flow_range'].float_list.value)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('invalid flow range')
            with np.load(path, allow_pickle=False) as arrays:
                records = background_arrays(arrays, flow, bounds)
            target = output/f'{index:02d}-background.npz'
            np.savez_compressed(target, **records, encoded_first_forward_flow=flow, forward_flow_range=bounds)
            errors = records['flow_error_pixels']
            result['scenes'].append(dict(name=name, file=target.name, sha256=digest(target),
                reference_sha256=digest(path), source_points=len(errors),
                flow_error_quantiles_pixels=np.quantile(errors, [0, .5, .95, 1]).tolist() if len(errors) else None,
                maximum_source_roundtrip_pixels=float(records['source_roundtrip_error_pixels'].max()) if len(errors) else None))
        if len(result['scenes']) != manifest['expected_scenes']:
            raise ValueError('incomplete fixed shard')
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'decoded', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.decoded, args.output)['scenes'], indent=2))
