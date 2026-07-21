"""Tests for the small shared dataset helpers in ``plumbline.datasets._common``
not already covered by ``test_ply_parser.py`` (manifests, env_path, RGB
image reading)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plumbline.datasets._common import env_path, load_manifest, read_rgb_uint8, save_manifest


class TestEnvPath:
    def test_env_var_set_returns_expanded_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLUMBLINE_TEST_VAR", "/tmp/some/dataset")
        assert env_path("PLUMBLINE_TEST_VAR") == Path("/tmp/some/dataset")

    def test_env_var_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PLUMBLINE_TEST_VAR_UNSET", raising=False)
        default = Path("/default/path")
        assert env_path("PLUMBLINE_TEST_VAR_UNSET", default) == default

    def test_env_var_unset_no_default_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PLUMBLINE_TEST_VAR_UNSET", raising=False)
        assert env_path("PLUMBLINE_TEST_VAR_UNSET") is None


class TestManifest:
    def test_roundtrip(self, tmp_path: Path) -> None:
        records = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        path = tmp_path / "manifest.jsonl"
        save_manifest(path, records)
        loaded = load_manifest(path)
        assert loaded == records

    def test_save_manifest_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "manifest.jsonl"
        save_manifest(path, [{"id": "a"}])
        assert path.exists()

    def test_load_manifest_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text('{"id": "a"}\n\n   \n{"id": "b"}\n', encoding="utf-8")
        records = load_manifest(path)
        assert records == [{"id": "a"}, {"id": "b"}]

    def test_load_manifest_malformed_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="malformed manifest at line 2"):
            load_manifest(path)


class TestReadRgbUint8:
    def test_reads_rgb_image(self, tmp_path: Path) -> None:
        from PIL import Image

        arr = (np.random.default_rng(0).random((4, 6, 3)) * 255).astype(np.uint8)
        path = tmp_path / "img.png"
        Image.fromarray(arr, mode="RGB").save(path)
        out = read_rgb_uint8(path)
        assert out.shape == (4, 6, 3)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, arr)

    def test_converts_grayscale_to_rgb(self, tmp_path: Path) -> None:
        from PIL import Image

        arr = (np.random.default_rng(1).random((4, 6)) * 255).astype(np.uint8)
        path = tmp_path / "gray.png"
        Image.fromarray(arr, mode="L").save(path)
        out = read_rgb_uint8(path)
        assert out.shape == (4, 6, 3)

    def test_converts_rgba_drops_alpha(self, tmp_path: Path) -> None:
        from PIL import Image

        arr = (np.random.default_rng(2).random((4, 6, 4)) * 255).astype(np.uint8)
        path = tmp_path / "rgba.png"
        Image.fromarray(arr, mode="RGBA").save(path)
        out = read_rgb_uint8(path)
        assert out.shape == (4, 6, 3)


class TestLoadPlyXyzListProperty:
    def test_list_property_raises_not_implemented(self, tmp_path: Path) -> None:
        from plumbline.datasets._common import load_ply_xyz

        # The list-property guard only fires for a property inside the
        # `vertex` element itself (per-vertex variable-length data, e.g. a
        # per-point neighbour list) — a list on a *different* element (like
        # `face`) is simply skipped, since only `vertex` properties are
        # collected.
        header = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property list uchar int neighbours\n"
            "end_header\n"
            "0.0 0.0 0.0 0\n"
        )
        path = tmp_path / "vertex_list_prop.ply"
        path.write_text(header, encoding="ascii")
        with pytest.raises(NotImplementedError, match="list properties"):
            load_ply_xyz(path)
