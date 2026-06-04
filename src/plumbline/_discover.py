"""Lazy discovery of built-in model + dataset adapters.

Adapters register themselves via decorators at module-import time. Users
interact via the registries, which live in
``plumbline.{models,datasets}.registry``. Importing *all* adapter modules
at package import time would pull in torch / transformers / h5py / etc.
unconditionally, which is undesirable for users who only want the CLI's
``list-models`` output.

This module provides a single discovery entry point so every caller
(CLI commands, ``run_reproduction``, programmatic API) wires the same
adapters with the same "missing deps are soft errors" behavior.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable

log = logging.getLogger(__name__)

_BUILTIN_ADAPTER_MODULES: tuple[str, ...] = (
    "plumbline.models.depth_anything_v2",
    "plumbline.models.metric3d_v2",
    "plumbline.models.mast3r",
    "plumbline.models.dust3r",
    "plumbline.models.vggt",
    "plumbline.models.depth_anything_3",
    "plumbline.models.moge",
    "plumbline.models.marigold",
    "plumbline.models.depth_pro",
    "plumbline.models.geowizard",
    "plumbline.models.cut3r",
    "plumbline.models.monst3r",
    "plumbline.models.dage",
    "plumbline.models.unik3d",
    "plumbline.models.vda",
    "plumbline.models.pi3",
    "plumbline.models.streamvggt",
    "plumbline.datasets.sintel",
    "plumbline.datasets.scannet",
    "plumbline.datasets.eth3d",
    "plumbline.datasets.nyuv2",
    "plumbline.datasets.kitti",
    "plumbline.datasets.diode",
    "plumbline.datasets.dtu",
    "plumbline.datasets.scannet_1500",
    "plumbline.datasets.co3dv2",
    "plumbline.datasets.co3dv2_vggt_eval",
    "plumbline.datasets.realestate10k_pose",
    "plumbline.datasets.gso",
    "plumbline.datasets.seven_scenes",
    "plumbline.datasets.ibims1",
    "plumbline.datasets.booster",
    "plumbline.datasets.eth3d_moge_eval",
    "plumbline.datasets.ddad_sintel_moge_eval",
    "plumbline.datasets.bonn",
    "plumbline.datasets.sun_rgbd_native",
    "plumbline.datasets.tum_dynamics",
    "plumbline.datasets.scannet_video_pose",
)


def register_builtin_adapters(*, raise_on_error: bool = False) -> list[tuple[str, Exception]]:
    """Import every built-in adapter module so decorators run.

    Returns a list of (module, exception) pairs for modules that failed to
    import (typically due to a missing optional dependency). When
    ``raise_on_error`` is True, the first failure propagates instead.

    Safe to call multiple times; Python caches module imports.
    """
    failures: list[tuple[str, Exception]] = []
    for mod in _BUILTIN_ADAPTER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            if raise_on_error:
                raise
            log.debug("skipping %s: %s", mod, exc)
            failures.append((mod, exc))
    return failures


def builtin_adapter_modules() -> Iterable[str]:
    """Return the canonical list of built-in adapter modules."""
    return _BUILTIN_ADAPTER_MODULES
