"""Shared path constants.

Kept in its own module so modules that can't import ``plumbline.reproduce``
(dataset loaders, etc., which ``reproduce`` itself imports from) can
resolve repo-relative resources without triggering an import cycle.

``protocols/`` and ``reproductions/`` live at the repo root for developer
ergonomics, but they are *runtime data* — ``load_protocol`` and
``_find_config`` read them on every reproduction run. A pip-installed
package has no repo root, so the wheel force-includes both trees under
``plumbline/`` (see ``[tool.hatch.build.targets.wheel.force-include]``) and
the resolution below prefers the checkout when present and falls back to
the packaged copy otherwise.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PROTOCOLS_DIR", "REPO_ROOT", "REPRODUCTIONS_DIR"]

_PACKAGE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _PACKAGE_DIR.parent.parent


def _resource_dir(name: str) -> Path:
    """Resolve a data dir, preferring the repo checkout over the packaged copy.

    Returns the repo-root path when it exists (editable installs and
    in-tree runs, where the checkout is the source of truth), else the copy
    bundled inside the package. Falls back to the repo-root path when
    neither exists so error messages still name the developer-facing
    location.
    """
    checkout = REPO_ROOT / name
    if checkout.is_dir():
        return checkout
    packaged = _PACKAGE_DIR / name
    if packaged.is_dir():
        return packaged
    return checkout


REPRODUCTIONS_DIR: Path = _resource_dir("reproductions")
PROTOCOLS_DIR: Path = _resource_dir("protocols")
