"""Reference-assisted depth observations; not reference-free evaluator labels."""

import numpy as np


def world_observation(xy, depth, intrinsics, camera_to_world):
    """Backproject nearest pixel with axial depth, retaining local discontinuities.

    Returns missing evidence explicitly. A large world-space change can reflect
    scene motion, a bad track, occlusion, depth noise, or calibration/timing error;
    it is not automatically a camera failure or a moving-object label.
    """
    if xy is None:
        return {"status": "missing_track"}
    xy = np.asarray(xy, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("invalid image point")
    if (
        k.shape != (3, 3)
        or pose.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(pose).all()
    ):
        raise ValueError("invalid camera")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("expected raw uint16 TUM depth image")
    x, y = np.rint(xy).astype(int)
    height, width = depth.shape
    if not (0 <= x < width and 0 <= y < height):
        return {"status": "outside_depth_image"}
    raw = int(depth[y, x])
    patch = depth[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)]
    valid = patch[patch > 0].astype(float) / 5000
    result = {
        "sampled_pixel_xy": [int(x), int(y)],
        "raw_depth": raw,
        "patch_valid_pixels": int(len(valid)),
        "patch_total_pixels": int(patch.size),
        "patch_depth_range_m": [float(valid.min()), float(valid.max())] if len(valid) else None,
    }
    if raw == 0:
        return {**result, "status": "missing_depth"}
    ray = np.linalg.solve(k, [x, y, 1.0])
    if abs(ray[2]) < 1e-12:
        raise ValueError("invalid axial-depth ray")
    camera_point = ray / ray[2] * (raw / 5000)
    world_point = pose[:3, :3] @ camera_point + pose[:3, 3]
    return {
        **result,
        "status": "measured",
        "depth_m": raw / 5000,
        "camera_point_m": camera_point.tolist(),
        "world_point_m": world_point.tolist(),
    }
