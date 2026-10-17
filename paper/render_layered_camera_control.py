"""Static three-depth pinhole control; reference geometry is scoring-only.

Separate from the frozen planar controls. Surface IDs are background=0,
left panel=1, right panel=2. No learned assets or real-video data are used.
"""

import numpy as np

from paper.render_camera_control import (
    HEIGHT, WIDTH, K, camera_pose, sample_texture, texture,
)

# Infinite rear wall and two bounded, non-coplanar static panels.
SURFACES = ((6., -np.inf, np.inf, -np.inf, np.inf),
            (3., -1.25, -.25, -.85, .85),
            (4., .25, 1.25, -.85, .85))


def intersections(origin, rays):
    """Nearest forward ray hit; shared scene definition, no camera estimates."""
    shape = rays.shape[:-1]
    distance = np.full(shape, np.inf)
    ids = np.full(shape, -1, dtype=int)
    hits = np.full((*shape, 3), np.nan)
    for sid, (z, xmin, xmax, ymin, ymax) in enumerate(SURFACES):
        t = np.divide(z-origin[2], rays[..., 2],
                      out=np.full(shape, np.inf), where=rays[..., 2] != 0)
        with np.errstate(invalid="ignore"):
            point = origin + t[..., None]*rays
        valid = ((t > 0) & (t < distance) & (point[..., 0] >= xmin)
                 & (point[..., 0] <= xmax) & (point[..., 1] >= ymin)
                 & (point[..., 1] <= ymax))
        distance[valid], ids[valid], hits[valid] = t[valid], sid, point[valid]
    return hits, ids


def render_frame(seed, frame, *, frames=24):
    pose = camera_pose(frame, frames)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1],
                     np.ones_like(xx)], -1) @ pose[:3, :3].T
    hits, ids = intersections(pose[:3, 3], rays)
    rgb = np.zeros((HEIGHT, WIDTH, 3))
    for sid in range(len(SURFACES)):
        mask = ids == sid
        rgb[mask] = sample_texture(texture(seed+sid*10000),
                                   hits[..., 0][mask], hits[..., 1][mask])
    return np.rint(rgb).clip(0, 255).astype(np.uint8), ids


def anchors():
    # Four points per surface, selected by geometry rather than tracking output.
    points = [[x, y, z] for z, xs in ((6., (-.2, .2)),
                                     (3., (-1., -.5)), (4., (.5, 1.)))
              for y in (-.45, .45) for x in xs]
    return np.array(points), np.repeat(np.arange(3), 4)


def project_anchors(points, surface_ids, frame, *, frames=24):
    """Scoring-only reference projections; analytic occlusion, no raster lookup."""
    points = np.asarray(points, dtype=float)
    surface_ids = np.asarray(surface_ids)
    if points.ndim != 2 or points.shape[1] != 3 or surface_ids.shape != (len(points),):
        raise ValueError("matching points and surface IDs required")
    for point, sid in zip(points, surface_ids):
        if sid not in range(3):
            raise ValueError("invalid surface ID")
        z, xmin, xmax, ymin, ymax = SURFACES[sid]
        if not (point[2] == z and xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
            raise ValueError("point is not on its declared surface")
    pose = camera_pose(frame, frames)
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    projected = camera @ K.T
    xy = projected[:, :2]/projected[:, 2:]
    _, nearest = intersections(pose[:3, 3], points-pose[:3, 3])
    visible = ((nearest == surface_ids) & (camera[:, 2] > 0)
               & (xy[:, 0] >= 0) & (xy[:, 0] <= WIDTH-1)
               & (xy[:, 1] >= 0) & (xy[:, 1] <= HEIGHT-1))
    return xy, visible
