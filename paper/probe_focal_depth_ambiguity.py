"""Known self-calibration ambiguity as a control on reference-free task claims."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.build_anchor_preview import fit_point, project
from paper.decode_movi_geometry_pilot import digest


def run():
    poses = np.tile(np.eye(4), (24, 1, 1))
    poses[:, 0, 3] = np.linspace(-.6, .6, 24)
    points = np.array([[-.5, -.3, 3.], [.2, .4, 4.], [.7, -.2, 6.], [-.1, .1, 8.]])
    size = np.array([256, 256])
    reference_k = np.tile([[1., 0, .5], [0, 1, .5], [0, 0, 1]], (24, 1, 1))
    observations = np.array([[project(point, pose, k, size) for pose, k in zip(poses, reference_k, strict=True)] for point in points])
    rows = []
    for factor in (.5, .8, 1., 1.25, 2.):
        calibration = reference_k.copy()
        calibration[:, 0, 0] *= factor
        calibration[:, 1, 1] *= factor
        alternative = points.copy(); alternative[:, 2] *= factor
        equivalent = np.array([[project(point, pose, k, size) for pose, k in zip(poses, calibration, strict=True)] for point in alternative])
        fitted = np.array([fit_point(track[:10], poses[:10], calibration[:10], size) for track in observations])
        forecast = np.array([[project(point, pose, k, size) for pose, k in zip(poses, calibration, strict=True)] for point in fitted])
        # The task's supplied metric world points are deliberately NOT rescaled.
        fixed_anchor = np.array([[project(point, pose, k, size) for pose, k in zip(poses, calibration, strict=True)] for point in points])
        rows.append(dict(focal_factor=factor, alternative_world_points=alternative.tolist(), fitted_world_points=fitted.tolist(),
            maximum_correspondence_difference_pixels=float(np.linalg.norm(equivalent-observations, axis=-1).max()),
            maximum_later_forecast_error_pixels=float(np.linalg.norm(forecast[:, 10:]-observations[:, 10:], axis=-1).max()),
            mean_known_anchor_task_error_pixels=float(np.linalg.norm(fixed_anchor[:, 10:]-observations[:, 10:], axis=-1).mean()),
            maximum_known_anchor_task_error_pixels=float(np.linalg.norm(fixed_anchor[:, 10:]-observations[:, 10:], axis=-1).max())))
    return dict(scope='analytic development control of a known calibration ambiguity, not a novel result or camera-quality benchmark',
        construction='24 parallel cameras translating along X, four static points at distinct depths; multiply both focal axes and all scene Z coordinates by the same positive factor; keep camera centers unchanged',
        derivation='u=f*(X-Cx)/Z+cx and v=f*Y/Z+cy are unchanged by f->a*f and Z->a*Z when camera Z=0 and orientations are identity',
        implication='point correspondence consistency cannot uniquely identify known-metric-world anchor error in this configuration without extra calibration or scene constraints; an RGB learned prior is not a geometric certificate',
        limitation='does not establish that any specific MOVi scene is exactly degenerate; does not prove RGB ambiguity for arbitrary appearance/lighting or camera motion',
        related_work='https://perception.inrialpes.fr/Publications/1999/Stu99/Sturm-bmvc99.pdf',
        script_sha256=digest(Path(__file__)), fit_helper_sha256=digest(Path(__file__).with_name('build_anchor_preview.py')),
        reference_world_points=points.tolist(), source_observations_pixels=observations.tolist(), cases=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run()
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['cases'], indent=2))
