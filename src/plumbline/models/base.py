"""Base ``Model`` ABC and ``Prediction`` dataclass.

Every model adapter subclasses :class:`Model`, owns all model-specific
preprocessing (device, dtype, resize, normalization), and converts the model's
native output into plumbline :mod:`~plumbline.conventions`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Model", "ModelCapabilities", "Prediction"]


@dataclass
class Prediction:
    """Canonical prediction from a model adapter for one sample (N views).

    All arrays are numpy, in canonical conventions. Set any field to ``None``
    when the model does not produce it; the runner skips the corresponding
    metric family.

    Attributes
    ----------
    depth
        ``(N, H, W)`` float32, meters when metric else dimensionless. NaN or 0
        denote invalid pixels.
    intrinsics
        ``(N, 3, 3)`` float32, pixels, in the input-image coordinate frame.
        If the adapter internally resized, it must unscale back to input-image
        pixels before returning.
    extrinsics
        ``(N, 4, 4)`` float32, ``world_from_camera`` with first camera as world
        origin.
    point_map
        ``(N, H, W, 3)`` float32 in the world frame.
    confidence
        ``(N, H, W)`` float32 in ``[0, 1]``. Higher = more certain. Runners
        ignore this for metric computation; report modules may use it.
    metadata
        Free-form dict for runtime stats (``runtime_ms``, ``peak_vram_mb``,
        checkpoint hash, etc.).
    """

    depth: NDArray[np.float32] | None = None
    intrinsics: NDArray[np.float32] | None = None
    extrinsics: NDArray[np.float32] | None = None
    point_map: NDArray[np.float32] | None = None
    confidence: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def has(self, field_name: str) -> bool:
        """Check whether this prediction provides ``field_name``."""
        return getattr(self, field_name, None) is not None


@dataclass(frozen=True)
class ModelCapabilities:
    """Declaration of what a model adapter supports.

    Attributes
    ----------
    tasks
        Task identifiers this model can be evaluated on, e.g.
        ``{"mono_depth", "mvs_depth", "pose"}``.
    is_metric
        Whether the model predicts metric (meters) depth. If False, scale
        alignment is required for metric comparison.
    min_views, max_views
        View-count bounds. Use ``math.inf`` for unbounded.
    requires_intrinsics
        If True, the adapter needs GT intrinsics passed to ``predict``; if
        False, the adapter predicts its own.
    default_resolution
        ``(H, W)`` resolution the model was trained at. The adapter may resize
        internally.
    """

    tasks: frozenset[str]
    is_metric: bool
    min_views: int = 1
    max_views: float = math.inf
    requires_intrinsics: bool = False
    default_resolution: tuple[int, int] | None = None

    def supports_task(self, task: str) -> bool:
        return task in self.tasks


class Model(ABC):
    """Abstract base for all plumbline model adapters.

    Adapter implementation rules (see :doc:`plan`):

    - Own the device, dtype, resize, normalization, and all model-specific
      preprocessing. The caller never deals with torch.
    - Convert native output to canonical conventions. Document every
      flip/transpose/scale with a comment citing the source.
    - Declare capabilities. If a model can't do pose, return
      ``Prediction.extrinsics=None`` and exclude ``"pose"`` from capabilities.
    - Keep weights in ``~/.cache/plumbline/weights/<model>/``; do not re-upload.
    """

    name: str
    version: str = "0.0.0"
    capabilities: ModelCapabilities

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Intentionally minimal: concrete subclasses may be abstract themselves
        # (e.g. an intermediate base for mono-depth models). Adapter registry
        # enforces presence of name/capabilities at registration time.

    @abstractmethod
    def predict(
        self,
        images: NDArray[np.uint8],
        intrinsics: NDArray[np.float32] | None = None,
    ) -> Prediction:
        """Run inference on ``N`` views.

        Parameters
        ----------
        images
            ``(N, H, W, 3)`` uint8 sRGB. ``H`` and ``W`` may vary across calls
            but are uniform within a call.
        intrinsics
            Optional ``(N, 3, 3)`` pixel-space intrinsics. Required when
            ``capabilities.requires_intrinsics`` is True.

        Returns
        -------
        Prediction
            With whichever fields the model provides.
        """

    @classmethod
    def from_hub(cls, name: str, device: str = "cuda") -> Model:
        """Instantiate a registered adapter by name."""
        from plumbline.models.registry import MODEL_REGISTRY

        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
        # Concrete adapters accept a `device` kwarg in __init__; the ABC
        # doesn't declare it to keep the interface flexible per-adapter.
        return MODEL_REGISTRY[name](device=device)  # type: ignore[call-arg]

    # Optional: override when the model has tunable preprocessing config.
    def config_hash(self) -> str:
        """Return a stable hash of model + preprocessing config for caching.

        Default: combination of name + version. Adapters that expose tunable
        knobs (resolution, view count, precision) must include them.
        """
        import hashlib

        return hashlib.sha256(f"{self.name}@{self.version}".encode()).hexdigest()[:16]
