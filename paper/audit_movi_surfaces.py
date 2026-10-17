"""Fixed-grid surface consistency audit; references only, no evaluator inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.decode_movi_geometry_pilot import digest


def object_to_world(coordinates, corners):
    """Kubric bbox order is itertools.product(x_bounds,y_bounds,z_bounds)."""
    origin = corners[...,0,:]
    edges = np.stack([corners[...,i,:]-origin for i in (4,2,1)],axis=-2)
    return origin + np.einsum('...i,...ij->...j', coordinates, edges)


def summarize(values):
    a = np.asarray(values)
    return dict(count=int(a.size), median=float(np.median(a)) if a.size else None,
                p95=float(np.quantile(a,.95)) if a.size else None,
                max=float(a.max()) if a.size else None)


def audit_arrays(a):
    segmentation = a['segmentations']
    t,h,w = segmentation.shape
    if (h,w) != (256,256):
        raise ValueError('fixed native-resolution pilot required')
    rotation = Rotation.from_quat(a['camera_quaternions'][:,[1,2,3,0]]).as_matrix()
    f = float(a['camera_focal_length'][0]/a['camera_sensor_width'][0])
    low,high = a['metadata_depth_range'].astype(float)
    if not (0 < low < high):
        raise ValueError('invalid radial depth range')
    columns = {k:[] for k in ('frame','x','y','instance','interior','projection_error_pixels','radial_depth_error','positive_depth')}
    for frame in range(t):
        yy,xx = np.meshgrid(np.arange(4,h,8),np.arange(4,w,8),indexing='ij')
        ids = segmentation[frame,yy,xx]
        keep = ids > 0
        y,x,ids = yy[keep],xx[keep],ids[keep].astype(int)
        if np.any(ids > len(a['instances_bboxes_3d'])):
            raise ValueError('unknown instance ID')
        coordinates = a['object_coordinates'][frame,y,x].astype(float)/65535
        world = object_to_world(coordinates,a['instances_bboxes_3d'][ids-1,frame])
        local = (world-a['camera_positions'][frame]) @ rotation[frame]
        if not np.isfinite(local).all() or np.any(np.abs(local[:,2]) < 1e-10):
            raise ValueError('nonfinite or undefined surface projection')
        uv = np.stack([.5-f*local[:,0]/local[:,2], .5+f*local[:,1]/local[:,2]],axis=-1)
        error = np.linalg.norm(uv*[w,h]-np.stack([x+.5,y+.5],axis=-1),axis=-1)
        depth = a['depth'][frame,y,x].astype(float)/65535*(high-low)+low
        radial_error = np.abs(np.linalg.norm(local,axis=-1)-depth)
        interior = np.ones(len(ids),dtype=bool)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                interior &= segmentation[frame,y+dy,x+dx] == ids
        for key,value in dict(frame=np.full(len(ids),frame),x=x,y=y,instance=ids,interior=interior,
                              projection_error_pixels=error,radial_depth_error=radial_error,positive_depth=local[:,2]<0).items():
            columns[key].extend(value.tolist())
    return {k:np.asarray(v) for k,v in columns.items()}


def audit(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or len(manifest['scenes']) != manifest['expected_scenes']:
        raise ValueError('complete pilot required')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(scope='surface metadata consistency, not temporal visibility or camera quality',
                  protocol='all 24 frames; fixed grid x,y=4,12,...,252; every foreground sample retained; 3x3 same-instance interior reported separately; object coordinates /65535; radial depth decoded with recorded min/max; pixel centers at x+.5,y+.5',
                  source_manifest_sha256=digest(root/'manifest.json'),script_sha256=digest(Path(__file__)),scenes=[])
    for scene in manifest['scenes']:
        path = root/scene['reference_file']
        if digest(path) != scene['reference_sha256']:
            raise ValueError('reference changed')
        with np.load(path,allow_pickle=False) as arrays:
            records = audit_arrays(arrays)
        target = output/f"{scene['index']:02d}-residuals.npz"
        np.savez_compressed(target,**records)
        interior = records['interior'].astype(bool)
        report['scenes'].append(dict(name=scene['name'],records_file=target.name,records_sha256=digest(target),
            planned_grid_samples=24*32*32,foreground_samples=len(interior),interior_samples=int(interior.sum()),
            nonpositive_depth=int((~records['positive_depth'].astype(bool)).sum()),
            projection_pixels=summarize(records['projection_error_pixels']),
            interior_projection_pixels=summarize(records['projection_error_pixels'][interior]),
            radial_depth_error=summarize(records['radial_depth_error']),
            interior_radial_depth_error=summarize(records['radial_depth_error'][interior])))
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,args.output)['scenes'],indent=2))
