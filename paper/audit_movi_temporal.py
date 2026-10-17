"""Continuous temporal reference diagnostics; no binary visibility or DNF labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.audit_movi_camera_metadata import project_centers
from paper.audit_movi_surfaces import object_to_world, summarize
from paper.decode_movi_geometry_pilot import digest


def temporal_arrays(a):
    masks = a['segmentations']
    frames,height,width = masks.shape
    yy,xx = np.meshgrid(np.arange(4,height,8),np.arange(4,width,8),indexing='ij')
    ids = masks[0,yy,xx]
    keep = ids > 0
    y,x,ids = yy[keep],xx[keep],ids[keep].astype(int)
    coordinates = a['object_coordinates'][0,y,x].astype(float)/65535
    world = object_to_world(coordinates[:,None,:],a['instances_bboxes_3d'][ids-1])
    fixed = np.broadcast_to(world[:,0:1],world.shape)
    focal_ratio = float(a['camera_focal_length'][0]/a['camera_sensor_width'][0])
    uv = project_centers(world,a['camera_positions'],a['camera_quaternions'],focal_ratio)*[width,height]
    uv_fixed = project_centers(fixed,a['camera_positions'],a['camera_quaternions'],focal_ratio)*[width,height]
    radial = np.linalg.norm(world-a['camera_positions'][None],axis=-1)
    rotation = Rotation.from_quat(a['camera_quaternions'][:,[1,2,3,0]]).as_matrix()
    local = np.einsum('tji,ntj->nti',rotation,world-a['camera_positions'][None])
    positive = local[...,2] < 0
    inside = positive&(uv[...,0] >= .5)&(uv[...,0] <= width-.5)&(uv[...,1] >= .5)&(uv[...,1] <= height-.5)
    depth_low,depth_high = a['metadata_depth_range'].astype(float)
    depth = a['depth'].astype(float)/65535*(depth_high-depth_low)+depth_low
    minimum,maximum = np.full(radial.shape,np.nan),np.full(radial.shape,np.nan)
    matching = np.zeros(radial.shape,dtype=np.uint8)
    for t in range(frames):
        slots = np.flatnonzero(inside[:,t])
        xy = np.floor(uv[slots,t]-.5).astype(int)
        neighbors,segments = [],[]
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            nx = np.clip(xy[:,0]+dx,0,width-1); ny = np.clip(xy[:,1]+dy,0,height-1)
            neighbors.append(depth[t,ny,nx]); segments.append(masks[t,ny,nx]==ids[slots])
        minimum[slots,t],maximum[slots,t] = np.min(neighbors,axis=0),np.max(neighbors,axis=0)
        matching[slots,t] = np.sum(segments,axis=0)
    return dict(source_xy=np.stack([x+.5,y+.5],axis=-1),instance=ids,
                initial_is_dynamic=a['initial_is_dynamic'][ids-1],
                world_displacement=np.linalg.norm(world-world[:,0:1],axis=-1),
                counterfactual_static_error_pixels=np.linalg.norm(uv-uv_fixed,axis=-1),
                projected_xy=uv,radial_depth=radial,positive_depth=positive,in_image=inside,
                neighbor_depth_min=minimum,neighbor_depth_max=maximum,matching_neighbor_count=matching,
                source_reprojection_error_pixels=np.linalg.norm(uv[:,0]-np.stack([x+.5,y+.5],axis=-1),axis=-1))


def audit(root,output):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or len(manifest['scenes']) != manifest['expected_scenes']:
        raise ValueError('complete fixed pilot required')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(scope='continuous reference-motion and local visibility evidence, not binary visibility or DNF labels',
        protocol='source frame 0; all foreground grid pixels x,y=4,12,...,252; fixed local object coordinate transported through every released bbox; compare actual position with source-frozen world point under the same reference camera; retain all frames and all points',
        limitations='four-neighbor depth and mask evidence is not exact visibility; surface normals and self-occlusion not yet independently checked; no staticness cutoff; source foreground only, no background support',
        source_manifest_sha256=digest(root/'manifest.json'),script_sha256=digest(Path(__file__)),scenes=[])
    for scene in manifest['scenes']:
        path = root/scene['reference_file']
        if digest(path) != scene['reference_sha256']:
            raise ValueError('reference changed')
        with np.load(path,allow_pickle=False) as arrays:
            records = temporal_arrays(arrays)
        dest = output/f"{scene['index']:02d}-temporal.npz"
        np.savez_compressed(dest,**records)
        inside = records['in_image']
        initially_still = records['initial_is_dynamic'] == 0
        report['scenes'].append(dict(name=scene['name'],records_file=dest.name,records_sha256=digest(dest),
            source_points=len(inside),planned_point_frames=int(inside.size),in_image=int(inside.sum()),
            source_reprojection_pixels=summarize(records['source_reprojection_error_pixels']),
            per_point_max_motion=summarize(records['world_displacement'].max(axis=1)),
            initially_still_max_motion=summarize(records['world_displacement'][initially_still].max(axis=1)),
            counterfactual_static_error_pixels=summarize(records['counterfactual_static_error_pixels']),
            neighbor_match_histogram={str(k):int(((records['matching_neighbor_count']==k)&inside).sum()) for k in range(5)}))
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,args.output)['scenes'],indent=2))
