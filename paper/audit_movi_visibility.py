"""Record depth, mask, and normal visibility evidence without claiming exact labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.audit_movi_surfaces import object_to_world
from paper.decode_movi_geometry_pilot import digest


def normal_evidence(source_normals, object_wxyz, world, camera_positions):
    n,t,_ = object_wxyz.shape
    rotation = Rotation.from_quat(object_wxyz.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(n,t,3,3)
    local = np.einsum('nji,nj->ni',rotation[:,0],source_normals)
    transported = np.einsum('ntij,nj->nti',rotation,local)
    dots = np.sum(transported*(world-camera_positions[None]),axis=-1)
    return dots, dots[:,0] <= 0


def evidence(a, normals, temporal):
    xy = (temporal['source_xy']-.5).astype(int)
    x,y = xy[:,0],xy[:,1]
    ids = temporal['instance'].astype(int)
    normal_map = normals.astype(float)/65535*2-1
    source = normal_map[0,y,x]
    rough = np.zeros(len(x),dtype=int)
    for dy,dx in ((-1,-1),(-1,1),(1,-1),(1,1)):
        rough += np.sum((normal_map[0,y+dy,x+dx]-source)**2,axis=-1) > .05**2
    trust = rough <= 2
    coordinates = a['object_coordinates'][0,y,x].astype(float)/65535
    world = object_to_world(coordinates[:,None],a['instances_bboxes_3d'][ids-1])
    dots, source_front = normal_evidence(source,a['instances_quaternions'][ids-1],world,a['camera_positions'])
    usable_normals = trust & source_front
    inside = temporal['in_image']
    depth_occluded = inside & (temporal['neighbor_depth_max'] < temporal['radial_depth']*.99)
    mask_occluded = inside & (temporal['matching_neighbor_count'] == 0)
    normal_occluded = usable_normals[:,None] & (dots > 0)
    rejected = ~inside | depth_occluded | mask_occluded | normal_occluded
    unresolved = ~rejected & ~usable_normals[:,None]
    supported = ~rejected & ~unresolved
    return dict(source_normal=source,normal_camera_dot=dots,source_front=source_front,
                normal_neighborhood_trusted=trust,normal_usable=usable_normals,
                depth_rejected=depth_occluded,mask_rejected=mask_occluded,normal_rejected=normal_occluded,
                outside_or_behind=~inside,rejected=rejected,unresolved=unresolved,supported=supported)


def audit(decoded,normals_root,temporal_root,output):
    dm,nm,tm = [json.loads((p/'manifest.json').read_text()) for p in (decoded,normals_root,temporal_root)]
    if (dm['status'] != 'complete' or nm['status'] != 'complete'
            or nm['decoded_manifest_sha256'] != digest(decoded/'manifest.json')
            or tm['source_manifest_sha256'] != digest(decoded/'manifest.json')
            or [s['name'] for s in dm['scenes']] != [s['name'] for s in nm['scenes']]
            or [s['name'] for s in dm['scenes']] != [s['name'] for s in tm['scenes']]):
        raise ValueError('pilot inputs do not match')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(scope='reference visibility evidence audit, not independently validated ground truth or camera DNF',
        policy='upstream-style four-neighbor max depth <0.99 projected radial depth, no matching instance, or trustworthy transported back-facing normal rejects; outside/behind rejects; normals with >2 diagonal neighbors differing by >0.05 or back-facing at source are unresolved unless another cue rejects',
        differences='fixed source grid and conservative image-center bounds; uncertain normals remain unresolved instead of being ignored; not an exact reproduction of upstream sampling/visibility',
        upstream='https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py',
        source_sha256={name:digest(root/'manifest.json') for name,root in (('decoded',decoded),('normals',normals_root),('temporal',temporal_root))},
        script_sha256=digest(Path(__file__)),scenes=[])
    for d,n,t in zip(dm['scenes'],nm['scenes'],tm['scenes']):
        paths = [decoded/d['reference_file'],normals_root/n['file'],temporal_root/t['records_file']]
        for path,sha in zip(paths,[d['reference_sha256'],n['sha256'],t['records_sha256']]):
            if digest(path) != sha:
                raise ValueError('input arrays changed')
        with np.load(paths[0],allow_pickle=False) as a, np.load(paths[1],allow_pickle=False) as normals, np.load(paths[2],allow_pickle=False) as temporal:
            records = evidence(a,normals['normal'],temporal)
        target = output/f"{d['index']:02d}-visibility.npz"
        np.savez_compressed(target,**records)
        counts = {k:int(records[k].sum()) for k in ('supported','rejected','unresolved','depth_rejected','mask_rejected','normal_rejected','outside_or_behind')}
        assert sum(counts[k] for k in ('supported','rejected','unresolved')) == records['supported'].size
        report['scenes'].append(dict(name=d['name'],records_file=target.name,records_sha256=digest(target),
            planned_point_frames=int(records['supported'].size),usable_source_normals=int(records['normal_usable'].sum()),counts=counts))
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded','normals','temporal','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.decoded,args.normals,args.temporal,args.output)['scenes'],indent=2))
