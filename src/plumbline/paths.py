"""Shared path constants.

Kept in its own module so modules that can't import ``plumbline.reproduce``
(dataset loaders, etc., which ``reproduce`` itself imports from) can
resolve repo-relative resources without triggering an import cycle.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["REPO_ROOT", "REPRODUCTIONS_DIR", "PROTOCOLS_DIR"]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
REPRODUCTIONS_DIR: Path = REPO_ROOT / "reproductions"
PROTOCOLS_DIR: Path = REPO_ROOT / "protocols"
