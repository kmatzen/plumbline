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
import importlib.metadata as importlib_metadata
import logging
from collections.abc import Iterable

log = logging.getLogger(__name__)

#: Entry-point group third-party packages use to register out-of-tree adapters
#: without forking plumbline. See :func:`load_plugin_adapters`.
ADAPTER_ENTRY_POINT_GROUP = "plumbline.adapters"

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
    "plumbline.models.vggt_omega",
    "plumbline.datasets.sintel",
    "plumbline.datasets.eth3d",
    "plumbline.datasets.eth3d_native_depth",
    "plumbline.datasets.nyuv2",
    "plumbline.datasets.nyu_cut3r_eval",
    "plumbline.datasets.kitti",
    "plumbline.datasets.diode",
    "plumbline.datasets.dtu",
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
)


def load_plugin_adapters(*, raise_on_error: bool = False) -> list[tuple[str, Exception]]:
    """Import third-party adapters advertised via the ``plumbline.adapters``
    entry-point group.

    This is what makes plumbline extensible **without cloning the repo**. Any
    installed distribution can ship its own :class:`~plumbline.models.base.Model`
    or :class:`~plumbline.datasets.base.Dataset` adapters and register them by
    declaring, in its own ``pyproject.toml``::

        [project.entry-points."plumbline.adapters"]
        my_adapters = "my_package.adapters"

    plumbline imports each such target at discovery time so the module's
    ``@register_model`` / ``@register_dataset`` decorators run. If an entry
    point points at a callable (``my_package.adapters:setup``), it is called
    after import. Failures (e.g. a plugin's missing optional dep) are soft by
    default, mirroring the built-in behavior, so one broken plugin can't block
    the harness.

    Returns a list of (entry-point label, exception) pairs for plugins that
    failed to load. Safe to call multiple times.
    """
    failures: list[tuple[str, Exception]] = []
    try:
        eps = importlib_metadata.entry_points(group=ADAPTER_ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - defensive (malformed metadata)
        if raise_on_error:
            raise
        log.debug("entry-point discovery failed: %s", exc)
        return failures
    for ep in eps:
        try:
            obj = ep.load()  # imports the target module -> decorators run
            if callable(obj):
                obj()
        except Exception as exc:
            if raise_on_error:
                raise
            label = f"{ADAPTER_ENTRY_POINT_GROUP}:{ep.name}"
            log.debug("skipping plugin %s: %s", label, exc)
            failures.append((label, exc))
    return failures


def register_builtin_adapters(*, raise_on_error: bool = False) -> list[tuple[str, Exception]]:
    """Import every built-in adapter module **and** any third-party plugins so
    their registration decorators run.

    Returns a list of (module-or-plugin, exception) pairs for targets that
    failed to import (typically a missing optional dependency). When
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
    # Out-of-tree plugins last, so a builtin name can't be shadowed by a plugin
    # (register_* raises on duplicate names).
    failures.extend(load_plugin_adapters(raise_on_error=raise_on_error))
    return failures


def builtin_adapter_modules() -> Iterable[str]:
    """Return the canonical list of built-in adapter modules."""
    return _BUILTIN_ADAPTER_MODULES
