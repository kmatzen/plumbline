"""Tests for ``run_reproduction`` — the backend of ``plumbline reproduce``.

Uses a temporary reproductions directory (via monkeypatching
``REPRODUCTIONS_DIR``), plus a fake model + fake dataset registered for the
duration of the test, so nothing leaks into the real registries.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.reproduce import ReproductionResult, load_reproduction_config, run_reproduction

# ---------------------------------------------------------------------------
# Test fixtures (fake model + fake dataset)
# ---------------------------------------------------------------------------


class _FixedDepthModel(Model):
    """Fake model that returns a constant AbsRel = ``target_abs_rel``."""

    capabilities = ModelCapabilities(
        tasks=frozenset({"mono_depth"}),
        is_metric=True,
        min_views=1,
        max_views=math.inf,
    )

    def __init__(self, *, device: str = "cpu", target_abs_rel: float = 0.10) -> None:
        self.device = device
        # To produce abs_rel=t deterministically: pred = gt * (1 - t) for constant t.
        self.target_abs_rel = float(target_abs_rel)

    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> Prediction:
        n, h, w, _ = images.shape
        # gt will be 1.0 in our fake dataset; (1 - t) * 1.0 gives |1 - (1-t)|/1 = t.
        depth = np.full((n, h, w), 1.0 - self.target_abs_rel, dtype=np.float32)
        return Prediction(depth=depth)


class _FakeDataset(Dataset):
    split = "test"

    def __init__(self, *, n_samples: int = 3) -> None:
        self.n = n_samples

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(
                sample_id=f"s{i}",
                images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                intrinsics=np.eye(3, dtype=np.float32)[None],
                extrinsics_gt=np.eye(4, dtype=np.float32)[None],
                depth_gt=np.ones((1, 4, 4), dtype=np.float32),
            )

    def __len__(self) -> int:
        return self.n


@pytest.fixture
def registered_fakes() -> Iterator[None]:
    """Register the fakes in MODEL_REGISTRY and DATASET_REGISTRY for one test."""
    model_name = "test-fixed-depth"
    dataset_name = "test-synthetic"
    before_models = dict(MODEL_REGISTRY)
    before_datasets = dict(DATASET_REGISTRY)
    _FixedDepthModel.name = model_name  # type: ignore[attr-defined]
    _FakeDataset.name = dataset_name  # type: ignore[attr-defined]
    MODEL_REGISTRY[model_name] = _FixedDepthModel
    DATASET_REGISTRY[dataset_name] = _FakeDataset
    try:
        yield
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(before_models)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(before_datasets)


@pytest.fixture
def repro_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect reproductions discovery + prediction cache to a tmp dir.

    The cache redirect matters: ``run_reproduction`` uses the default cache
    under ``~/.cache/plumbline`` if the YAML doesn't set one, which would
    leak predictions across tests (cache key doesn't include model kwargs
    by default).
    """
    d = tmp_path / "reproductions"
    d.mkdir()
    monkeypatch.setattr("plumbline.reproduce.REPRODUCTIONS_DIR", d)
    monkeypatch.setenv("PLUMBLINE_CACHE_DIR", str(tmp_path / "cache"))
    return d


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
