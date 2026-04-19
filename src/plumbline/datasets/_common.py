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
    "load_ply_xyz",
    "read_rgb_uint8",
    "save_manifest",
]


_PLY_PROP_BYTES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}  # fmt: skip


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


def load_ply_xyz(path: Path) -> NDArray[np.float32]:
    """Minimal PLY parser: returns ``(N, 3)`` float32 XYZ only.

    Supports ``ascii`` and ``binary_little_endian`` with ``float`` XYZ as the
    first three vertex properties. Computes the vertex stride from the
    header so files with extra elements (e.g. ETH3D's ``scan_clean`` PLYs
    that append a trailing ``element camera`` block, or DTU's GT clouds
    that ship as plain vertex-only PLY) don't mislead the reshape.
    """
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line.startswith("end_header"):
                break
        fmt = next(
            (ln.split()[1] for ln in header_lines if ln.startswith("format")),
            "ascii",
        )
        # Vertex element + its property widths. Ignore any later elements.
        vcount = 0
        vertex_props: list[str] = []
        in_vertex = False
        for ln in header_lines:
            if ln.startswith("element "):
                parts = ln.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vcount = int(parts[2])
            elif in_vertex and ln.startswith("property "):
                parts = ln.split()
                if parts[1] == "list":
                    raise NotImplementedError("list properties unsupported")
                vertex_props.append(parts[1])
        payload = f.read()

    if fmt.startswith("binary_little_endian"):
        vertex_stride = sum(_PLY_PROP_BYTES[p] for p in vertex_props)
        vertex_bytes = vcount * vertex_stride
        buf = np.frombuffer(payload[:vertex_bytes], dtype=np.uint8).reshape(vcount, vertex_stride)
        xyz = np.frombuffer(buf[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
        return np.ascontiguousarray(xyz)

    xyz = np.empty((vcount, 3), dtype=np.float32)
    for i, line in enumerate(payload.decode("ascii").splitlines()[:vcount]):
        parts = line.split()
        xyz[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return xyz


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
