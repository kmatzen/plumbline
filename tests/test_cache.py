"""Tests for the prediction cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plumbline.cache import PredictionCache, default_cache_dir, sanitize_id
from plumbline.models.base import Prediction


def _make_prediction(seed: int = 0) -> Prediction:
    rng = np.random.default_rng(seed)
    return Prediction(
        depth=rng.random((2, 4, 4), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(2, axis=0),
        metadata={"runtime_ms": 12.3, "checkpoint": "abcd1234"},
    )


class TestSanitize:
    def test_plain(self) -> None:
        assert sanitize_id("scene_0000_view_01") == "scene_0000_view_01"

    def test_unsafe_chars(self) -> None:
        assert sanitize_id("scene/00 00:0 7") == "scene_00_00_0_7"

    def test_long_id_truncated(self) -> None:
        long = "a" * 200
        out = sanitize_id(long)
        assert len(out) <= 128
        # Different long IDs with the same prefix should still differ.
        out2 = sanitize_id("a" * 199 + "b")
        assert out != out2

    def test_empty_fallback(self) -> None:
        assert sanitize_id("") == "_"


class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        pred = _make_prediction(1)
        path = cache.save("m", "cfg", "ds", "s0", pred)
        assert path.exists()
        loaded = cache.load("m", "cfg", "ds", "s0")
        np.testing.assert_array_equal(loaded.depth, pred.depth)
        np.testing.assert_array_equal(loaded.intrinsics, pred.intrinsics)
        assert loaded.metadata == pred.metadata

    def test_has(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        assert not cache.has("m", "cfg", "ds", "s0")
        cache.save("m", "cfg", "ds", "s0", _make_prediction())
        assert cache.has("m", "cfg", "ds", "s0")

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        import pytest

        with pytest.raises(FileNotFoundError):
            cache.load("m", "cfg", "ds", "nope")

    def test_missing_fields_stay_none(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        pred = Prediction(depth=np.zeros((1, 2, 2), dtype=np.float32))
        cache.save("m", "cfg", "ds", "s0", pred)
        out = cache.load("m", "cfg", "ds", "s0")
        assert out.depth is not None
        assert out.intrinsics is None
        assert out.extrinsics is None

    def test_numpy_scalar_metadata(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        pred = Prediction(metadata={"peak_vram_mb": np.float32(512.5)})
        cache.save("m", "cfg", "ds", "s0", pred)
        out = cache.load("m", "cfg", "ds", "s0")
        assert out.metadata["peak_vram_mb"] == 512.5

    def test_atomic_write_tmp_cleanup(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        cache.save("m", "cfg", "ds", "s0", _make_prediction())
        tmp_files = list(tmp_path.rglob("*.npz.tmp"))
        assert tmp_files == []


class TestClear:
    def test_clear_all(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        cache.save("m1", "cfg", "ds", "s0", _make_prediction())
        cache.save("m2", "cfg", "ds", "s0", _make_prediction())
        removed = cache.clear()
        assert removed == 2
        assert not cache.has("m1", "cfg", "ds", "s0")

    def test_clear_one_model(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        cache.save("m1", "cfg", "ds", "s0", _make_prediction())
        cache.save("m2", "cfg", "ds", "s0", _make_prediction())
        removed = cache.clear(model="m1")
        assert removed == 1
        assert not cache.has("m1", "cfg", "ds", "s0")
        assert cache.has("m2", "cfg", "ds", "s0")

    def test_clear_one_dataset_across_models(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        cache.save("m1", "cfg", "ds_a", "s0", _make_prediction())
        cache.save("m1", "cfg", "ds_b", "s0", _make_prediction())
        cache.save("m2", "cfg", "ds_a", "s0", _make_prediction())
        removed = cache.clear(dataset="ds_a")
        assert removed == 2
        assert cache.has("m1", "cfg", "ds_b", "s0")
        assert not cache.has("m1", "cfg", "ds_a", "s0")
        assert not cache.has("m2", "cfg", "ds_a", "s0")

    def test_clear_nothing_when_empty(self, tmp_path: Path) -> None:
        cache = PredictionCache(tmp_path)
        assert cache.clear() == 0


class TestDefaultCacheDir:
    def test_respects_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PLUMBLINE_CACHE_DIR", str(tmp_path / "custom"))
        assert default_cache_dir() == tmp_path / "custom"

    def test_falls_back_to_home(self, monkeypatch) -> None:
        monkeypatch.delenv("PLUMBLINE_CACHE_DIR", raising=False)
        d = default_cache_dir()
        assert d.name == "plumbline"
