"""Procedural pinhole control with exact surface identities, not a real-data benchmark.

RGB is rendered by ray/plane intersection. Reference coordinates and visibility
are returned separately and must never enter a reference-free evaluator.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


WIDTH, HEIGHT = 320, 240
K = np.array([[260., 0., 159.5], [0., 260., 119.5], [0., 0., 1.]])


def camera_pose(frame: int, frames: int = 24) -> np.ndarray:
    if frames < 2 or not 0 <= frame < frames:
        raise ValueError("valid frame in a sequence of at least two frames required")
    phase = frame / (frames - 1)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", 3 * phase, degrees=True).as_matrix()
    pose[0, 3] = .5 * phase
    return pose


def texture(seed: int) -> np.ndarray:
    # Fixed local randomness; no learned data or recognizable semantic objects.
    rng = np.random.default_rng(seed)
    return rng.integers(35, 230, (256, 256, 3), dtype=np.uint8)


def sample_texture(tex, x, y):
    # Periodic bilinear texture in world units, shared by all views.
    u, v = (x * 32) % 256, (y * 32) % 256
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    a, b = (u - i)[..., None], (v - j)[..., None]
    return ((1-a)*(1-b)*tex[j, i] + a*(1-b)*tex[j, (i+1) % 256]
            + (1-a)*b*tex[(j+1) % 256, i] + a*b*tex[(j+1) % 256, (i+1) % 256])


def render_frame(seed: int, frame: int, *, frames: int = 24, occluder: bool = True):
    pose = camera_pose(frame, frames)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)], -1)
    rays = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    # Infinite static background at z=4, finite moving foreground at z=2.5.
    background = origin + rays * ((4-origin[2])/rays[..., 2])[..., None]
    rgb = sample_texture(texture(seed), background[..., 0], background[..., 1])
    surface = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if occluder:
        foreground = origin + rays * ((2.5-origin[2])/rays[..., 2])[..., None]
        center = -1.4 + 2.8 * frame/(frames-1)
        mask = ((np.abs(foreground[..., 0]-center) <= .32)
                & (np.abs(foreground[..., 1]) <= .65))
        rgb[mask] = sample_texture(texture(seed+10000), foreground[..., 0][mask]-center,
                                  foreground[..., 1][mask])
        surface[mask] = 1
    return np.rint(rgb).clip(0, 255).astype(np.uint8), surface


def project_anchors(points, frame, *, frames=24, occluder=True):
    """Analytic projection and visibility, independent of raster rounding."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(points[:, 2] == 4):
        raise ValueError("background points at z=4 required")
    pose = camera_pose(frame, frames)
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    pixels = camera @ K.T
    xy = pixels[:, :2] / pixels[:, 2:]
    visible = ((camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= WIDTH-1)
               & (xy[:, 1] >= 0) & (xy[:, 1] <= HEIGHT-1))
    if occluder:
        origin = pose[:3, 3]
        hit = origin + (points-origin) * ((2.5-origin[2])/(points[:, 2]-origin[2]))[:, None]
        center = -1.4 + 2.8 * frame/(frames-1)
        visible &= ~((np.abs(hit[:, 0]-center) <= .32) & (np.abs(hit[:, 1]) <= .65))
    return xy, visible


def anchors():
    return np.array([[x, y, 4.] for y in (-.7, 0., .7) for x in (-.9, -.3, .3, .9)])
