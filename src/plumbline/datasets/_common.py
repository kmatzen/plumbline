"""Shared helpers for dataset loaders: manifests, path resolution, errors."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DatasetNotAvailable",
    "env_path",
    "load_manifest",
    "read_rgb_uint8",
    "save_manifest",
]


class DatasetNotAvailable(RuntimeError):  # noqa: N818 - keep domain-specific name
    """Raised when a required dataset is not present on disk.

    Message must include the expected path layout and (when relevant) the
    download URL with an instruction to place files there.
    """


def env_path(var: str, default: Path | None = None) -> Path | None:
    """Read a path from ``$var`` or fall back to ``default``."""
    val = os.environ.get(var)
    if val:
        return Path(val).expanduser()
    return default


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a newline-delimited JSON manifest."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest at line {i}: {exc}") from exc
    return records


def save_manifest(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")


def read_rgb_uint8(path: Path) -> NDArray[np.uint8]:
    """Read an image as ``(H, W, 3)`` uint8 sRGB, dropping alpha if present."""
    from PIL import Image

    with Image.open(path) as raw:
        if raw.mode in ("L", "RGBA") or raw.mode not in ("RGB", "RGBA", "L"):
            img = raw.convert("RGB")
        else:
            img = raw
        arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) from {path}, got shape {arr.shape}")
    return arr
