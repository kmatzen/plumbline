"""Tests for the ``plumbline make-samples`` CLI command."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from plumbline.cli import app
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import DATASET_REGISTRY


class _FakeDataset(Dataset):
    split = "test"

    def __init__(self, *, n_samples: int = 12) -> None:
        self.n = n_samples

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(
                sample_id=f"sample_{i:04d}",
                images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                intrinsics=np.eye(3, dtype=np.float32)[None],
                extrinsics_gt=np.eye(4, dtype=np.float32)[None],
            )

    def __len__(self) -> int:
        return self.n


@pytest.fixture
def registered_fake() -> Iterator[str]:
    name = "test-samples-ds"
    before = dict(DATASET_REGISTRY)
    _FakeDataset.name = name  # type: ignore[attr-defined]
    DATASET_REGISTRY[name] = _FakeDataset
    try:
        yield name
    finally:
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(before)


runner = CliRunner()


def _parse_ids(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_writes_all_sample_ids(tmp_path: Path, registered_fake: str) -> None:
    output = tmp_path / "samples.txt"
    result = runner.invoke(
        app,
        ["make-samples", "--dataset", registered_fake, "-o", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    ids = _parse_ids(output.read_text())
    assert ids == [f"sample_{i:04d}" for i in range(12)]


def test_subset_limits_output(tmp_path: Path, registered_fake: str) -> None:
    output = tmp_path / "samples.txt"
    result = runner.invoke(
        app,
        ["make-samples", "--dataset", registered_fake, "--subset", "3", "-o", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    ids = _parse_ids(output.read_text())
    # linspace(0, 11, 3).round() = [0, 6, 11].
    assert ids == ["sample_0000", "sample_0006", "sample_0011"]


def test_header_includes_metadata(tmp_path: Path, registered_fake: str) -> None:
    output = tmp_path / "samples.txt"
    runner.invoke(
        app,
        ["make-samples", "--dataset", registered_fake, "--subset", "5", "-o", str(output)],
    )
    text = output.read_text()
    # Header is comments at the top.
    head = [line for line in text.splitlines() if line.startswith("#")]
    joined = "\n".join(head)
    assert "plumbline" in joined
    assert f"dataset: {registered_fake}" in joined
    assert "subset: 5" in joined
    assert "n_samples: 5" in joined


def test_unknown_dataset_errors(tmp_path: Path) -> None:
    output = tmp_path / "samples.txt"
    result = runner.invoke(
        app,
        ["make-samples", "--dataset", "no-such-dataset", "-o", str(output)],
    )
    assert result.exit_code != 0
    # BadParameter from typer surfaces via output+exception.
    combined = result.output + (str(result.exception) if result.exception else "")
    assert "Unknown dataset" in combined or "no-such-dataset" in combined
    assert not output.exists()


def test_output_is_round_trippable_by_subset_by_ids(tmp_path: Path, registered_fake: str) -> None:
    """The file written by make-samples must be consumable by subset_by_ids."""
    from plumbline.reproduce import _read_sample_ids

    output = tmp_path / "samples.txt"
    runner.invoke(
        app,
        ["make-samples", "--dataset", registered_fake, "--subset", "4", "-o", str(output)],
    )

    ids = _read_sample_ids(output)
    ds = _FakeDataset(n_samples=12)
    pinned = ds.subset_by_ids(ids)
    got = [s.sample_id for s in pinned]
    assert got == ids
