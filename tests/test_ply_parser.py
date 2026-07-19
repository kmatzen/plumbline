"""Guards on the minimal PLY reader.

``load_ply_xyz`` reinterprets the first 12 bytes of each vertex record as
three float32s. When that assumption is wrong the stride arithmetic still
works, so a bad parse yields *plausible garbage* rather than an error — and a
chamfer/F-score eval runs to completion against a scrambled GT cloud. These
tests pin the checks that turn silent corruption into a loud failure.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from plumbline.datasets._common import load_ply_xyz

_POINTS = [(1.0, 2.0, 3.0), (-4.5, 5.5, 6.25), (0.0, 0.0, 1.0)]


def _write_ply(
    path: Path,
    *,
    fmt: str = "binary_little_endian",
    xyz_type: str = "float",
    extra_props: tuple[str, ...] = (),
    prop_names: tuple[str, str, str] = ("x", "y", "z"),
) -> None:
    lines = ["ply", f"format {fmt} 1.0", f"element vertex {len(_POINTS)}"]
    lines += [f"property {xyz_type} {n}" for n in prop_names]
    lines += [f"property uchar {n}" for n in extra_props]
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")

    if fmt == "ascii":
        body = "".join(f"{x} {y} {z}\n" for x, y, z in _POINTS).encode("ascii")
    else:
        pack = "<3f" if xyz_type == "float" else "<3d"
        body = b"".join(struct.pack(pack, *p) + bytes(len(extra_props)) for p in _POINTS)
    path.write_bytes(header + body)


class TestLoadPlyXyz:
    def test_reads_binary_little_endian(self, tmp_path: Path) -> None:
        p = tmp_path / "cloud.ply"
        _write_ply(p)
        np.testing.assert_allclose(load_ply_xyz(p), np.array(_POINTS, dtype=np.float32))

    def test_reads_ascii(self, tmp_path: Path) -> None:
        p = tmp_path / "cloud.ply"
        _write_ply(p, fmt="ascii")
        np.testing.assert_allclose(load_ply_xyz(p), np.array(_POINTS, dtype=np.float32))

    def test_trailing_properties_do_not_shift_xyz(self, tmp_path: Path) -> None:
        """Extra per-vertex bytes (colors) must widen the stride, not the read."""
        p = tmp_path / "colored.ply"
        _write_ply(p, extra_props=("red", "green", "blue"))
        np.testing.assert_allclose(load_ply_xyz(p), np.array(_POINTS, dtype=np.float32))

    def test_double_xyz_raises_instead_of_returning_garbage(self, tmp_path: Path) -> None:
        p = tmp_path / "double.ply"
        _write_ply(p, xyz_type="double")
        with pytest.raises(NotImplementedError, match="float32 x/y/z"):
            load_ply_xyz(p)

    def test_non_xyz_leading_properties_raise(self, tmp_path: Path) -> None:
        p = tmp_path / "normals_first.ply"
        _write_ply(p, prop_names=("nx", "ny", "nz"))
        with pytest.raises(NotImplementedError, match="x, y, z"):
            load_ply_xyz(p)

    def test_big_endian_raises_clearly(self, tmp_path: Path) -> None:
        """Previously fell through to the ASCII branch and died on a decode error."""
        p = tmp_path / "be.ply"
        _write_ply(p, fmt="binary_big_endian")
        with pytest.raises(NotImplementedError, match="binary_big_endian"):
            load_ply_xyz(p)
