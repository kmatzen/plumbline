"""Tests for ``run_reproduction`` — the backend of ``plumbline reproduce``.

Uses a temporary reproductions directory (via monkeypatching
``REPRODUCTIONS_DIR``), plus a fake model + fake dataset registered for the
duration of the test, so nothing leaks into the real registries. The fake
classes live in ``tests/_fakes.py`` and the registration fixtures
(``registered_fakes``, ``registered_pointmap_fakes``, ``repro_dir``) are
auto-discovered from ``tests/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from plumbline.datasets.base import Sample
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.reproduce import ReproductionResult, load_reproduction_config, run_reproduction
from tests._fakes import _MultiViewPointCloudDataset


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# load_reproduction_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_basic(self, repro_dir: Path) -> None:
        _write_yaml(repro_dir / "fake.yaml", "name: fake\nmodel: {name: m}\n")
        cfg = load_reproduction_config("fake")
        assert cfg["name"] == "fake"
        assert cfg["model"]["name"] == "m"

    def test_underscore_fallback(self, repro_dir: Path) -> None:
        """Hyphens in the name map to underscored filenames too."""
        _write_yaml(repro_dir / "x_y_z.yaml", "name: x\n")
        cfg = load_reproduction_config("x-y-z")
        assert cfg["name"] == "x"

    def test_not_found(self, repro_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No reproduction config"):
            load_reproduction_config("nope")


# ---------------------------------------------------------------------------
# run_reproduction
# ---------------------------------------------------------------------------


class TestBuiltinDiscovery:
    def test_run_reproduction_auto_discovers_builtin_adapters(
        self, repro_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: ``plumbline reproduce`` must auto-register built-ins.

        Previously, calling ``run_reproduction`` on a YAML that pointed at a
        built-in adapter (e.g. ``depth-anything-v2``) would fail with
        ``KeyError: model '...' not registered`` because nothing imported the
        adapter module. Now ``run_reproduction`` calls
        ``register_builtin_adapters()`` first.
        """
        _write_yaml(
            repro_dir / "r.yaml",
            """
name: r
model: {name: depth-anything-v2, kwargs: {variant: small, device: cpu}}
dataset: {name: nyuv2, kwargs: {root: /nonexistent/path}}
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        # Should raise DatasetNotAvailable (missing root) — NOT KeyError on
        # model registration. Reaching the dataset construction path proves
        # both model + dataset auto-discovery ran.
        from plumbline.datasets._common import DatasetNotAvailable

        with pytest.raises((DatasetNotAvailable, FileNotFoundError)):
            run_reproduction("r")


class TestRunReproduction:
    def test_match_within_tolerance(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model:
  name: test-fixed-depth
  kwargs:
    target_abs_rel: 0.10
dataset:
  name: test-synthetic
  kwargs:
    n_samples: 3
tasks:
  - mono_depth
scale_alignment: none
max_views: 1
device: cpu
paper_reference:
  primary_metric: abs_rel
  value: 0.10
  tolerance_relative: 0.05
""".strip(),
        )
        result = run_reproduction("repro")
        assert isinstance(result, ReproductionResult)
        assert result.primary_metric == "abs_rel"
        assert result.observed == pytest.approx(0.10, abs=1e-6)
        assert result.published == pytest.approx(0.10)
        assert result.paper_match is True
        assert result.report.n_evaluated == 3

    def test_mismatch_outside_tolerance(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth, kwargs: {target_abs_rel: 0.20}}
dataset: {name: test-synthetic, kwargs: {n_samples: 2}}
tasks: [mono_depth]
scale_alignment: none
device: cpu
paper_reference:
  primary_metric: abs_rel
  value: 0.10
  tolerance_relative: 0.05
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.paper_match is False

    def test_unknown_model(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: no-such-model}
dataset: {name: test-synthetic}
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        with pytest.raises(KeyError, match="no-such-model"):
            run_reproduction("repro")

    def test_unknown_dataset(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: no-such-dataset}
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        with pytest.raises(KeyError, match="no-such-dataset"):
            run_reproduction("repro")

    def test_subset_applied(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: test-synthetic, kwargs: {n_samples: 10}}
subset: 3
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.report.n_evaluated == 3

    def test_no_paper_value_leaves_match_none(
        self, repro_dir: Path, registered_fakes: None
    ) -> None:
        """If the YAML omits the paper value, match check is informational."""
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: test-synthetic}
tasks: [mono_depth]
device: cpu
paper_reference:
  primary_metric: abs_rel
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.paper_match is None

    def test_report_saved_when_output_given(
        self, repro_dir: Path, tmp_path: Path, registered_fakes: None
    ) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: test-synthetic}
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        out = tmp_path / "out.json"
        run_reproduction("repro", output=out)
        assert out.exists()

    def test_sample_ids_file_pins_exact_samples(
        self, repro_dir: Path, registered_fakes: None
    ) -> None:
        """`sample_ids_file` pins the exact sample set, not a stride."""
        (repro_dir / "pinned.txt").write_text("s1\ns3\n# comment\n\ns4\n")
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: test-synthetic, kwargs: {n_samples: 10}}
sample_ids_file: pinned.txt
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.report.n_evaluated == 3
        ids = [s.sample_id for s in result.report.per_sample]
        assert ids == ["s1", "s3", "s4"]

    def test_sample_ids_file_missing_id_raises(
        self, repro_dir: Path, registered_fakes: None
    ) -> None:
        (repro_dir / "pinned.txt").write_text("s1\nno_such_sample\n")
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth}
dataset: {name: test-synthetic, kwargs: {n_samples: 3}}
sample_ids_file: pinned.txt
tasks: [mono_depth]
device: cpu
""".strip(),
        )
        with pytest.raises(KeyError, match="were not found"):
            run_reproduction("repro")

    def test_min_samples_shortfall_forces_no_match(
        self, repro_dir: Path, registered_fakes: None
    ) -> None:
        """A count below ``min_samples`` must fail even if the metric matches.

        Guards the D28 footgun: the metric lands exactly on paper (0.10) but
        only 3 samples were evaluated against a declared floor of 5, so the
        run was on the wrong set and must not count as a paper match.
        """
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth, kwargs: {target_abs_rel: 0.10}}
dataset: {name: test-synthetic, kwargs: {n_samples: 3}}
tasks: [mono_depth]
scale_alignment: none
max_views: 1
device: cpu
min_samples: 5
paper_reference:
  primary_metric: abs_rel
  value: 0.10
  tolerance_relative: 0.05
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.observed == pytest.approx(0.10, abs=1e-6)  # metric matches
        assert result.paper_match is False  # but forced no-match
        assert result.count_shortfall is True
        assert result.n_evaluated == 3
        assert result.min_samples == 5
        assert "COUNT SHORTFALL" in result.notes
        assert "BELOW MINIMUM" in result.to_markdown()

    def test_min_samples_met_matches_normally(
        self, repro_dir: Path, registered_fakes: None
    ) -> None:
        """When the floor is met the gate behaves exactly as without it."""
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth, kwargs: {target_abs_rel: 0.10}}
dataset: {name: test-synthetic, kwargs: {n_samples: 4}}
tasks: [mono_depth]
scale_alignment: none
max_views: 1
device: cpu
min_samples: 3
paper_reference:
  primary_metric: abs_rel
  value: 0.10
  tolerance_relative: 0.05
""".strip(),
        )
        result = run_reproduction("repro")
        assert result.paper_match is True
        assert result.count_shortfall is False
        assert result.n_evaluated == 4

    def test_to_markdown_renders_status(self, repro_dir: Path, registered_fakes: None) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-fixed-depth, kwargs: {target_abs_rel: 0.10}}
dataset: {name: test-synthetic}
tasks: [mono_depth]
device: cpu
paper_reference:
  primary_metric: abs_rel
  value: 0.10
  tolerance_relative: 0.05
""".strip(),
        )
        md = run_reproduction("repro").to_markdown()
        assert "Reproduction check" in md
        assert "abs_rel" in md
        assert "Match:" in md


class TestChamferReproduction:
    """Regression: ``pointcloud_alignment`` must flow from YAML → runner.

    These tests are what the ETH3D chamfer reproduction YAMLs exercise in
    production. Without this coverage, silently dropping
    ``pointcloud_alignment=cfg.get(...)`` in reproduce.py would make every
    chamfer YAML run *without* the 7-DoF similarity fit, scoring random
    chamfer against the GT frame — the exact failure mode that 7-DoF
    alignment was added to prevent.
    """

    def test_mvs_depth_task_fires_chamfer_and_f_score(
        self, repro_dir: Path, registered_pointmap_fakes: None
    ) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-pointmap}
dataset: {name: test-pointcloud, kwargs: {n_samples: 2}}
tasks: [mvs_depth]
pointcloud_alignment: camera_centers
max_views: 3
device: cpu
paper_reference:
  primary_metric: chamfer
""".strip(),
        )
        result = run_reproduction("repro")
        metrics = result.report.aggregate_metrics
        # The chamfer path must fire whenever tasks include mvs_depth AND the
        # prediction has point_map AND the sample has point_cloud_gt.
        assert "chamfer" in metrics
        assert "f_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        # Finite. Exact value depends on GT spread vs predicted point map;
        # identity camera layout here means Umeyama is the identity so
        # chamfer equals the GT self-spread, not a near-zero match.
        assert np.isfinite(metrics["chamfer"])
        assert metrics["chamfer"] < 1.0

    def test_camera_centers_alignment_beats_none(
        self, repro_dir: Path, registered_pointmap_fakes: None
    ) -> None:
        """With a deliberate world-frame offset, Umeyama alignment should
        dramatically reduce chamfer vs running without alignment."""

        # Shadow the point-cloud dataset with one whose GT cameras are offset
        # by +10 m from the predicted extrinsics. Without alignment, pred
        # point map lives at Z=1 near origin; GT is centred ~10 m away.
        class _OffsetGTDataset(_MultiViewPointCloudDataset):
            def __iter__(self) -> Iterator[Sample]:
                for base in super().__iter__():
                    ext = base.extrinsics_gt.copy()
                    ext[:, 0, 3] += 10.0
                    pcd = base.point_cloud_gt.copy() if base.point_cloud_gt is not None else None
                    if pcd is not None:
                        pcd[:, 0] += 10.0
                    yield Sample(
                        sample_id=base.sample_id,
                        images=base.images,
                        intrinsics=base.intrinsics,
                        extrinsics_gt=ext,
                        point_cloud_gt=pcd,
                    )

        _OffsetGTDataset.name = "test-pointcloud-offset"  # type: ignore[attr-defined]
        DATASET_REGISTRY["test-pointcloud-offset"] = _OffsetGTDataset
        try:
            for mode in ("none", "camera_centers"):
                _write_yaml(
                    repro_dir / f"repro_{mode}.yaml",
                    f"""
name: repro_{mode}
model: {{name: test-pointmap}}
dataset: {{name: test-pointcloud-offset, kwargs: {{n_samples: 1}}}}
tasks: [mvs_depth]
pointcloud_alignment: {mode}
max_views: 3
device: cpu
paper_reference:
  primary_metric: chamfer
""".strip(),
                )
            unaligned = run_reproduction("repro_none").report.aggregate_metrics["chamfer"]
            aligned = run_reproduction("repro_camera_centers").report.aggregate_metrics["chamfer"]
            assert unaligned > 5.0, f"expected large chamfer without alignment; got {unaligned}"
            assert aligned < 1.0, f"expected small chamfer with alignment; got {aligned}"
            assert aligned < unaligned
        finally:
            DATASET_REGISTRY.pop("test-pointcloud-offset", None)

    def test_unknown_pointcloud_alignment_raises(
        self, repro_dir: Path, registered_pointmap_fakes: None
    ) -> None:
        _write_yaml(
            repro_dir / "repro.yaml",
            """
name: repro
model: {name: test-pointmap}
dataset: {name: test-pointcloud}
tasks: [mvs_depth]
pointcloud_alignment: not-a-real-mode
device: cpu
""".strip(),
        )
        with pytest.raises(ValueError, match="unknown pointcloud_alignment"):
            run_reproduction("repro")


class TestBundledReproductionFilenames:
    """Every bundled reproduction YAML must be discoverable via its own slug.

    Regression: a YAML whose internal ``name:`` slug differs from its
    filename stem lives at a path that ``plumbline reproduce <slug>``
    can't find. Any future drift
    where the internal ``name:`` and the filename stem stop matching
    makes the config silently uninvokable, which is exactly the
    failure mode this test catches.
    """

    def test_every_bundled_yaml_is_invokable_by_its_name(self) -> None:
        import yaml as yaml_mod  # local import so test module stays lightweight

        from plumbline.reproduce import REPRODUCTIONS_DIR, load_reproduction_config

        yamls = sorted(REPRODUCTIONS_DIR.glob("*.yaml"))
        assert yamls, "expected at least one bundled reproduction YAML"
        checked = 0
        for path in yamls:
            cfg = yaml_mod.safe_load(path.read_text(encoding="utf-8"))
            # Non-reproduction configs (e.g. gpu_queue.yaml) live in the same
            # directory but have no `model:` block — skip them; they're not
            # invoked via `plumbline reproduce`.
            if not isinstance(cfg, dict) or "model" not in cfg:
                continue
            assert "name" in cfg, f"{path.name} is missing a top-level 'name:' field"
            checked += 1
            slug = cfg["name"]
            # load_reproduction_config accepts both hyphen- and underscore-
            # separated slugs; if it can't find either, the YAML is
            # effectively orphaned from `plumbline reproduce`.
            try:
                load_reproduction_config(slug)
            except FileNotFoundError as exc:
                raise AssertionError(
                    f"{path.name} declares name={slug!r} but that slug does "
                    f"not resolve to a YAML under {REPRODUCTIONS_DIR}. "
                    f"Fix by renaming the file to match the slug "
                    f"(hyphens → underscores)."
                ) from exc
        assert checked >= 20, f"expected to check >= 20 bundled reproduction YAMLs, got {checked}"
