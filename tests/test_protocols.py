"""Tests for the protocol-preset merge layer.

Covers :func:`plumbline.protocols.apply_protocol` — the function that
merges a protocol YAML's ``fixed`` fields into a reproduction config
and raises if the reproduction tries to override a fixed field.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plumbline.paths import PROTOCOLS_DIR
from plumbline.protocols import (
    ProtocolConflictError,
    apply_protocol,
    load_protocol,
)


class TestLoadProtocol:
    def test_loads_shipped_nyu_preset(self) -> None:
        cfg = load_protocol("nyu_eigen_2014")
        assert cfg["name"] == "nyu_eigen_2014"
        assert "fixed" in cfg
        assert cfg["fixed"]["dataset"]["name"] == "nyuv2"
        assert cfg["fixed"]["depth_clip"] == [0.001, 10.0]

    def test_loads_shipped_eth3d_dav2_preset(self) -> None:
        cfg = load_protocol("eth3d_dav2")
        assert cfg["fixed"]["dataset"]["name"] == "eth3d"
        assert cfg["fixed"]["dataset"]["kwargs"]["views_per_sample"] == 1
        assert cfg["fixed"]["dataset"]["kwargs"]["with_per_view_gt"] is True
        assert cfg["fixed"]["dataset"]["kwargs"]["resize_images_to_pv_render"] is True
        assert cfg["fixed"]["depth_clip"] == [0.001, 80.0]

    def test_loads_shipped_sintel_dav2_preset(self) -> None:
        cfg = load_protocol("sintel_dav2")
        assert cfg["fixed"]["dataset"]["kwargs"]["max_depth"] == 70.0
        assert cfg["fixed"]["dataset"]["kwargs"]["pass_name"] == "final"

    def test_missing_protocol_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_protocol("does-not-exist")

    def test_preset_without_fixed_block_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "bad.yaml").write_text("name: bad\n")
        with pytest.raises(ValueError, match="fixed"):
            load_protocol("bad")


class TestApplyProtocol:
    def test_noop_when_no_protocol_declared(self) -> None:
        cfg = {"name": "x", "model": {"name": "m"}}
        out = apply_protocol(cfg)
        assert out == cfg

    def test_merges_fixed_into_reproduction(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "p.yaml").write_text(yaml.safe_dump({
            "name": "p",
            "fixed": {
                "dataset": {"name": "d", "kwargs": {"a": 1}},
                "depth_clip": [0.0, 10.0],
                "tasks": ["mono_depth"],
            },
        }))
        repro = {
            "protocol": "p",
            "model": {"name": "m"},
            "scale_alignment": "median",
        }
        out = apply_protocol(repro)
        assert out["dataset"] == {"name": "d", "kwargs": {"a": 1}}
        assert out["depth_clip"] == [0.0, 10.0]
        assert out["tasks"] == ["mono_depth"]
        assert out["model"] == {"name": "m"}  # preserved
        assert out["scale_alignment"] == "median"  # preserved

    def test_raises_on_conflict_at_leaf(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "p.yaml").write_text(yaml.safe_dump({
            "name": "p",
            "fixed": {"depth_clip": [0.001, 10.0]},
        }))
        repro = {"protocol": "p", "depth_clip": [0.0, 80.0]}
        with pytest.raises(ProtocolConflictError, match="depth_clip"):
            apply_protocol(repro)

    def test_raises_on_conflict_at_nested_leaf(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "p.yaml").write_text(yaml.safe_dump({
            "name": "p",
            "fixed": {"dataset": {"kwargs": {"apply_eigen_crop": True}}},
        }))
        repro = {
            "protocol": "p",
            "dataset": {"kwargs": {"apply_eigen_crop": False}},
        }
        with pytest.raises(ProtocolConflictError, match="dataset.kwargs.apply_eigen_crop"):
            apply_protocol(repro)

    def test_redundant_matching_value_is_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the reproduction explicitly sets a fixed field to the same
        # value the protocol fixes, that's allowed — users may want to
        # keep the field visible in the YAML as self-documentation.
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "p.yaml").write_text(yaml.safe_dump({
            "name": "p",
            "fixed": {"depth_clip": [0.001, 10.0]},
        }))
        repro = {"protocol": "p", "depth_clip": [0.001, 10.0]}
        out = apply_protocol(repro)
        assert out["depth_clip"] == [0.001, 10.0]

    def test_conflict_error_lists_every_mismatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plumbline.protocols.PROTOCOLS_DIR", tmp_path)
        (tmp_path / "p.yaml").write_text(yaml.safe_dump({
            "name": "p",
            "fixed": {
                "depth_clip": [0.001, 10.0],
                "tasks": ["mono_depth"],
            },
        }))
        repro = {
            "protocol": "p",
            "depth_clip": [0.0, 80.0],
            "tasks": ["pose"],
        }
        with pytest.raises(ProtocolConflictError) as excinfo:
            apply_protocol(repro)
        msg = str(excinfo.value)
        assert "depth_clip" in msg
        assert "tasks" in msg


class TestShippedReproductionsLoadCleanly:
    """Every committed reproduction YAML should resolve under the
    current protocol presets without ProtocolConflictError. This
    catches drift from a protocol tweak breaking existing YAMLs.
    """

    def test_da_v2_small_nyuv2_loads_under_nyu_protocol(self) -> None:
        import yaml as yaml_mod

        from plumbline.paths import REPRODUCTIONS_DIR

        path = REPRODUCTIONS_DIR / "da_v2_small_nyuv2.yaml"
        with path.open() as f:
            cfg = yaml_mod.safe_load(f)
        assert cfg.get("protocol") == "nyu_eigen_2014"
        merged = apply_protocol(cfg)
        # Protocol-fixed fields now present:
        assert merged["dataset"]["name"] == "nyuv2"
        assert merged["dataset"]["kwargs"]["apply_eigen_crop"] is True
        assert merged["dataset"]["kwargs"]["depth_field"] == "raw"
        assert merged["depth_clip"] == [0.001, 10.0]
        assert merged["tasks"] == ["mono_depth"]
        assert merged["max_views"] == 1
        # Reproduction-specific fields preserved:
        assert merged["model"]["name"] == "depth-anything-v2"
        assert merged["model"]["kwargs"]["variant"] == "small"
        assert merged["scale_alignment"] == "scale_shift"

    def test_every_protocol_bearing_reproduction_resolves(self) -> None:
        """Sweeps every `reproductions/*.yaml` that declares a
        ``protocol:`` field and confirms ``apply_protocol`` resolves
        without conflict. Any YAML that silently drifts away from
        its protocol (e.g. re-adds a depth_clip that differs) shows
        up here as a :class:`ProtocolConflictError`.
        """
        import yaml as yaml_mod

        from plumbline.paths import REPRODUCTIONS_DIR

        checked = 0
        for path in sorted(REPRODUCTIONS_DIR.glob("*.yaml")):
            with path.open() as f:
                cfg = yaml_mod.safe_load(f)
            if not isinstance(cfg, dict) or not cfg.get("protocol"):
                continue
            # Will raise ProtocolConflictError on any drift.
            merged = apply_protocol(cfg)
            # Sanity: protocol-fixed fields are now present.
            assert "dataset" in merged
            assert "tasks" in merged
            assert "max_views" in merged
            checked += 1
        # We expect at least the NYU+KITTI sweep (22 YAMLs).
        assert checked >= 20, f"expected to check >= 20 protocol-bearing YAMLs, got {checked}"
