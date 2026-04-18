"""Tests for report JSON/markdown serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from plumbline.report import SCHEMA_VERSION, Report, RunEnvironment, SampleResult


def _example_report() -> Report:
    env = RunEnvironment(
        python="3.12.0",
        platform="test-platform",
        plumbline_version="0.1.0.dev0",
        torch_version="2.3.0",
        cuda_version="12.1",
        gpu_name="RTX-4090",
        timestamp_utc="2026-04-18T00:00:00+00:00",
    )
    return Report(
        model="fake-model",
        model_version="v1",
        dataset="fake-dataset",
        split="test",
        tasks=["mono_depth"],
        scale_alignment="median",
        aggregate_metrics={"abs_rel": 0.1234, "rmse": 0.5678},
        per_sample=[
            SampleResult("s0", metrics={"abs_rel": 0.1}, runtime_ms=42.0),
            SampleResult("s1", metrics={"abs_rel": 0.15}, runtime_ms=40.0),
            SampleResult("s2", metrics={}, skipped=True, skip_reason="OOM"),
        ],
        n_total=3,
        n_evaluated=2,
        n_skipped=1,
        environment=env,
        config_hash="deadbeef",
        notes="hello",
    )


class TestJson:
    def test_schema_version_present(self) -> None:
        r = _example_report()
        d = r.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION

    def test_round_trip(self, tmp_path: Path) -> None:
        r = _example_report()
        p = r.save_json(tmp_path / "report.json")
        loaded = Report.load_json(p)
        assert loaded.model == r.model
        assert loaded.aggregate_metrics == r.aggregate_metrics
        assert len(loaded.per_sample) == len(r.per_sample)
        assert loaded.per_sample[2].skipped is True
        assert loaded.environment.gpu_name == "RTX-4090"

    def test_rejects_unknown_schema(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"schema_version": "9.9.9", "model": "x", "dataset": "y"}')
        with pytest.raises(ValueError, match="schema version"):
            Report.load_json(p)


class TestMarkdown:
    def test_contains_headers_and_metrics(self) -> None:
        md = _example_report().to_markdown()
        assert "fake-model" in md
        assert "fake-dataset" in md
        assert "abs_rel" in md
        assert "median" in md
        assert "Samples evaluated:" in md and "2 / 3" in md
        assert "Skipped:" in md and " 1" in md

    def test_handles_nan(self) -> None:
        r = _example_report()
        r.aggregate_metrics["nan_metric"] = float("nan")
        md = r.to_markdown()
        assert "nan_metric" in md
        assert "—" in md
