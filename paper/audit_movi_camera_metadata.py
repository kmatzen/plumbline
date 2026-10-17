"""Check reference camera conventions against released object-center projections."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.decode_movi_geometry_pilot import digest


def project_centers(points, camera_positions, wxyz, focal_sensor_ratio):
    if (not np.isfinite(wxyz).all()
            or not np.allclose(np.linalg.norm(wxyz,axis=-1),1,atol=1e-6,rtol=0)):
        raise ValueError("invalid camera quaternion")
    rotation = Rotation.from_quat(wxyz[:,[1,2,3,0]]).as_matrix()
    local = np.einsum('tji,ntj->nti',rotation,points-camera_positions[None])
    if not np.isfinite(local).all() or np.any(np.abs(local[...,2]) < 1e-10):
        raise ValueError("undefined center projection")
    f = focal_sensor_ratio
    return np.stack([.5-f*local[...,0]/local[...,2], .5+f*local[...,1]/local[...,2]],axis=-1)


def audit(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or len(manifest['scenes']) != manifest['expected_scenes']:
        raise ValueError("complete fixed pilot required")
    scenes = []
    for scene in manifest['scenes']:
        path = root/scene['reference_file']
        if digest(path) != scene['reference_sha256']:
            raise ValueError("reference arrays changed")
        with np.load(path,allow_pickle=False) as arrays:
            uv = project_centers(arrays['instances_positions'], arrays['camera_positions'],
                arrays['camera_quaternions'],float(arrays['camera_focal_length'][0]/arrays['camera_sensor_width'][0]))
            error = np.linalg.norm((uv-arrays['instances_image_positions'])*256,axis=-1)
            scenes.append(dict(name=scene['name'],comparisons=int(error.size),
                max_pixels=float(error.max()),median_pixels=float(np.median(error)),errors_pixels=error.tolist()))
    return dict(scope="reference metadata projection consistency only; not depth, visibility, surface correspondence, or camera-quality validation",
                convention="wxyz camera-to-world quaternion; camera front -Z and up +Y; normalized pixel coordinates; square 256x256 pilot",
                source="https://github.com/google-research/kubric/blob/main/kubric/core/cameras.py",
                decoded_manifest_sha256=digest(root/'manifest.json'), script_sha256=digest(Path(__file__)),scenes=scenes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
