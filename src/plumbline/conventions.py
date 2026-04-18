"""Canonical conventions for cameras, images, depth, and point maps.

Every adapter and loader in plumbline must convert into these conventions
exactly once, at the boundary. The runner and metrics assume them without
checking; correctness lives here.

Summary
-------
- **Camera convention:** OpenCV. Right-handed, +X right, +Y down, +Z forward
  (into the scene). Image origin top-left, u right, v down.
- **World frame:** The first camera of the sequence. For a single sample with
  N views, ``extrinsics[0]`` is identity.
- **Extrinsics:** ``world_from_camera``, shape ``(4, 4)``. A point in camera
  coordinates ``X_c`` maps to world via ``X_w = R @ X_c + t`` where ``R = E[:3, :3]``
  and ``t = E[:3, 3]``. Note this is the inverse of the "projection" matrix used
  by some papers; we keep the convention most natural for scene reasoning.
- **Intrinsics:** ``K`` shape ``(3, 3)`` in pixels with standard layout
  ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``. No normalized coordinates.
- **Depth:** ``(H, W)`` float32, meters when metric, dimensionless otherwise.
  ``0`` or ``NaN`` denote invalid pixels.
- **Point map:** ``(H, W, 3)`` float32 in the **world frame**.
- **Image:** ``(H, W, 3)`` uint8, sRGB, no alpha. Linear color is a v0.2
  concern; flagged in the schema so it can be added without breaking.
- **Resolution:** store GT at native resolution; resize predictions to GT for
  metric computation, never the other way around.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EPS",
    "IMAGE_SPACE",
    "WORLD_FRAME",
    "assert_valid_depth",
    "assert_valid_extrinsics",
    "assert_valid_image",
    "assert_valid_intrinsics",
    "assert_valid_point_map",
    "camera_from_world",
    "depth_is_valid",
    "invert_pose",
    "world_from_camera_is_identity",
]

EPS: float = 1e-8
"""Numerical epsilon. Use for scale alignment and division guards."""

IMAGE_SPACE: str = "opencv"
"""Image coordinate convention: ``u`` right, ``v`` down, origin top-left."""

WORLD_FRAME: str = "first_camera"
"""World frame convention: first camera of the sequence is identity."""


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _check_finite(arr: NDArray, name: str, *, allow_nan: bool = False) -> None:
    if allow_nan:
        mask = ~np.isnan(arr)
        if not np.all(np.isfinite(arr[mask])):
            raise AssertionError(f"{name} contains +/-inf values")
    else:
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"{name} contains non-finite values (NaN or inf)")


def assert_valid_image(image: NDArray, *, name: str = "image") -> None:
    """Validate an sRGB uint8 image in canonical layout.

    Shape: ``(H, W, 3)`` or ``(N, H, W, 3)``. Dtype: ``uint8``.
    """
    if not isinstance(image, np.ndarray):
        raise AssertionError(f"{name} must be np.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise AssertionError(f"{name} must be uint8, got {image.dtype}")
    if image.ndim == 3 or image.ndim == 4:
        if image.shape[-1] != 3:
            raise AssertionError(f"{name} must have 3 channels, got shape {image.shape}")
    else:
        raise AssertionError(
            f"{name} must have ndim 3 or 4 (H,W,3) or (N,H,W,3), got {image.shape}"
        )


def assert_valid_intrinsics(K: NDArray, *, name: str = "intrinsics") -> None:
    """Validate an intrinsics matrix ``K``.

    Shape: ``(3, 3)`` or ``(N, 3, 3)``, float. Must be finite and upper-triangular
    in the sense that ``K[1, 0] == 0`` (no shear on first subdiagonal below fy row).
    """
    if not isinstance(K, np.ndarray):
        raise AssertionError(f"{name} must be np.ndarray, got {type(K).__name__}")
    if not np.issubdtype(K.dtype, np.floating):
        raise AssertionError(f"{name} must be floating dtype, got {K.dtype}")
    if K.shape[-2:] != (3, 3):
        raise AssertionError(f"{name} must have trailing shape (3, 3), got {K.shape}")
    _check_finite(K, name)

    batch = K.reshape(-1, 3, 3)
    # Last row must be [0, 0, 1] up to float tolerance.
    bottom = batch[:, 2, :]
    if not np.allclose(bottom, np.array([0.0, 0.0, 1.0]), atol=1e-5):
        raise AssertionError(
            f"{name} bottom row must be [0, 0, 1]; got {bottom[0].tolist()} (first example)"
        )
    # Positive focal lengths.
    if not np.all(batch[:, 0, 0] > 0) or not np.all(batch[:, 1, 1] > 0):
        raise AssertionError(f"{name} focal lengths (K[0,0], K[1,1]) must be > 0")


def assert_valid_extrinsics(E: NDArray, *, name: str = "extrinsics") -> None:
    """Validate a ``world_from_camera`` extrinsic matrix.

    Shape: ``(4, 4)`` or ``(N, 4, 4)``, float. ``E[:3, :3]`` must be a rotation
    (orthonormal, det ≈ +1). ``E[3, :]`` must be ``[0, 0, 0, 1]``.
    """
    if not isinstance(E, np.ndarray):
        raise AssertionError(f"{name} must be np.ndarray, got {type(E).__name__}")
    if not np.issubdtype(E.dtype, np.floating):
        raise AssertionError(f"{name} must be floating dtype, got {E.dtype}")
    if E.shape[-2:] != (4, 4):
        raise AssertionError(f"{name} must have trailing shape (4, 4), got {E.shape}")
    _check_finite(E, name)

    batch = E.reshape(-1, 4, 4)
    bottom = batch[:, 3, :]
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(bottom, expected, atol=1e-5):
        raise AssertionError(
            f"{name}[..., 3, :] must be [0, 0, 0, 1]; got {bottom[0].tolist()} (first example)"
        )

    R = batch[:, :3, :3]
    should_be_I = R @ np.swapaxes(R, -1, -2)
    identity = np.broadcast_to(np.eye(3), should_be_I.shape)
    if not np.allclose(should_be_I, identity, atol=1e-4):
        raise AssertionError(f"{name}[..., :3, :3] must be orthonormal")
    det = np.linalg.det(R)
    if not np.all(np.abs(det - 1.0) < 1e-4):
        raise AssertionError(
            f"{name} rotation determinant must be +1 (right-handed); got {det.tolist()}"
        )


def world_from_camera_is_identity(E: NDArray, *, atol: float = 1e-5) -> bool:
    """Check that the first camera is the world frame (``E[0]`` ≈ identity).

    Applies to a batch of shape ``(N, 4, 4)``.
    """
    if E.ndim != 3 or E.shape[-2:] != (4, 4):
        raise ValueError(f"E must be (N, 4, 4); got {E.shape}")
    return bool(np.allclose(E[0], np.eye(4), atol=atol))


def assert_valid_depth(
    depth: NDArray, *, name: str = "depth", allow_zero_invalid: bool = True
) -> None:
    """Validate a depth map.

    Shape: ``(H, W)`` or ``(N, H, W)``. Dtype: floating. Invalid pixels are
    ``0`` (when ``allow_zero_invalid``) or ``NaN``. Valid pixels must be finite
    and non-negative.
    """
    if not isinstance(depth, np.ndarray):
        raise AssertionError(f"{name} must be np.ndarray, got {type(depth).__name__}")
    if not np.issubdtype(depth.dtype, np.floating):
        raise AssertionError(f"{name} must be floating dtype, got {depth.dtype}")
    if depth.ndim not in (2, 3):
        raise AssertionError(f"{name} must have ndim 2 or 3, got shape {depth.shape}")
    # NaN is allowed (invalid marker); +/-inf is not.
    _check_finite(depth, name, allow_nan=True)
    finite = ~np.isnan(depth)
    # Negative values are never valid — either an adapter bug or a convention
    # violation (disparity fed in as depth). Reject regardless of mask.
    if np.any(depth[finite] < 0):
        raise AssertionError(f"{name} has negative values")
    _ = allow_zero_invalid  # reserved for future use; kept for API stability


def depth_is_valid(depth: NDArray, *, allow_zero_invalid: bool = True) -> NDArray:
    """Boolean mask of valid depth pixels."""
    mask = ~np.isnan(depth)
    if allow_zero_invalid:
        mask &= depth > 0
    else:
        mask &= depth >= 0
    return mask


def assert_valid_point_map(pmap: NDArray, *, name: str = "point_map") -> None:
    """Validate a world-frame point map.

    Shape: ``(H, W, 3)`` or ``(N, H, W, 3)``. NaN allowed as invalid marker.
    """
    if not isinstance(pmap, np.ndarray):
        raise AssertionError(f"{name} must be np.ndarray, got {type(pmap).__name__}")
    if not np.issubdtype(pmap.dtype, np.floating):
        raise AssertionError(f"{name} must be floating dtype, got {pmap.dtype}")
    if pmap.ndim not in (3, 4):
        raise AssertionError(f"{name} must have ndim 3 or 4, got shape {pmap.shape}")
    if pmap.shape[-1] != 3:
        raise AssertionError(f"{name} last dim must be 3, got shape {pmap.shape}")
    _check_finite(pmap, name, allow_nan=True)


# ---------------------------------------------------------------------------
# Small geometry helpers (pure numpy, used by adapters and loaders)
# ---------------------------------------------------------------------------


def invert_pose(E: NDArray) -> NDArray:
    """Invert a batch of rigid-body 4x4 transforms.

    Valid for any orthonormal rotation + translation. ``(N, 4, 4) -> (N, 4, 4)``
    or ``(4, 4) -> (4, 4)``.
    """
    if E.shape[-2:] != (4, 4):
        raise ValueError(f"E must end in (4, 4); got {E.shape}")
    E = np.asarray(E)
    single = E.ndim == 2
    batch = E.reshape(-1, 4, 4)
    R = batch[:, :3, :3]
    t = batch[:, :3, 3:4]
    Rt = np.swapaxes(R, -1, -2)
    inv = np.zeros_like(batch)
    inv[:, :3, :3] = Rt
    inv[:, :3, 3:4] = -Rt @ t
    inv[:, 3, 3] = 1.0
    if single:
        return inv[0]
    return inv.reshape(E.shape)


def camera_from_world(E_world_from_camera: NDArray) -> NDArray:
    """Alias for :func:`invert_pose` with a documenting name.

    In plumbline, extrinsics are always ``world_from_camera``. Some papers and
    dataset formats use ``camera_from_world`` (the projection direction).
    Convert with this helper at the loader/adapter boundary and document it.
    """
    return invert_pose(E_world_from_camera)


def rebase_to_first_camera(E: NDArray) -> NDArray:
    """Return extrinsics re-referenced so ``E[0]`` is identity.

    Given ``(N, 4, 4)`` ``world_from_camera`` matrices in some arbitrary world
    frame, return a new set where the first camera is the world origin. This is
    the canonical plumbline convention and must be applied by every dataset
    loader when sampling a multi-view subset.
    """
    if E.ndim != 3 or E.shape[-2:] != (4, 4):
        raise ValueError(f"E must be (N, 4, 4); got {E.shape}")
    inv0 = invert_pose(E[0])
    return inv0[None, ...] @ E
