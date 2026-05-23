"""Tests for the GPU run queue (``plumbline.queue``).

Two layers:

1. Validation against the *bundled* ``reproductions/gpu_queue.yaml`` — every
   job must point at a real reproduction, statuses must be valid, and the two
   verified pose targets must be present and pending. This is what stops the
   queue from silently rotting as reproductions are renamed/removed.
2. ``run_queue`` behaviour on a *synthetic* queue + reproductions in a tmp dir,
   so we can exercise match / mismatch / error outcomes without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from plumbline import queue as queue_mod

# The ``registered_fakes`` + ``repro_dir`` fixtures are auto-discovered from
# tests/conftest.py; jobs reference the fake model by its registered name
# ("test-fixed-depth"), so no direct class imports are needed here.


# ---------------------------------------------------------------------------
# Bundled queue validation
# ---------------------------------------------------------------------------


class TestBundledQueue:
    def test_loads_and_validates(self) -> None:
        jobs = queue_mod.load_queue()
        assert jobs, "bundled gpu_queue.yaml has no jobs"

    def test_sorted_by_priority(self) -> None:
        jobs = queue_mod.load_queue()
        priorities = [j.priority for j in jobs]
        assert priorities == sorted(priorities)

    def test_every_job_resolves_to_a_reproduction(self) -> None:
        from plumbline.reproduce import _find_config

        for job in queue_mod.load_queue():
            assert _find_config(job.name) is not None, (
                f"queue job {job.name!r} does not resolve to a reproduction YAML"
            )

    def test_verified_pose_targets_present_and_pending(self) -> None:
        """The two PDF-verified pose runs are the v0.1 release gate; they must
        stay in the queue and runnable until the GPU run lands."""
        by_name = {j.name: j for j in queue_mod.load_queue()}
        for name in ("vggt-co3dv2-pose", "mast3r-co3dv2-pose"):
            assert name in by_name, f"{name} dropped from the GPU queue"
            assert by_name[name].status == "pending"
            assert by_name[name].runnable

    def test_upstream_blocked_jobs_are_blocked_with_reason(self) -> None:
        """Upstream-blocked cells must be marked blocked (so they're never
        auto-run) and carry a blocked_on reason."""
        by_name = {j.name: j for j in queue_mod.load_queue()}
        for name in ("vggt-paper-dtu-mvs", "geowizard-nyuv2", "marigold-v1-1-kitti"):
            assert name in by_name
            assert by_name[name].status == "blocked"
            assert by_name[name].blocked_on.strip(), f"{name} blocked without a reason"
            assert not by_name[name].runnable


# ---------------------------------------------------------------------------
# Validation errors on malformed queues
# ---------------------------------------------------------------------------


class TestQueueValidation:
    def _write_queue(self, path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")

    def test_unknown_status_rejected(self, tmp_path: Path) -> None:
        q = tmp_path / "q.yaml"
        self._write_queue(q, "jobs:\n  - {name: foo, status: wat}\n")
        with pytest.raises(ValueError, match="invalid status"):
            queue_mod.load_queue(q)

    def test_unresolvable_name_rejected(self, tmp_path: Path) -> None:
        q = tmp_path / "q.yaml"
        self._write_queue(q, "jobs:\n  - {name: no-such-repro-xyz, status: pending}\n")
        with pytest.raises(ValueError, match="no reproduction config"):
            queue_mod.load_queue(q)

    def test_duplicate_name_rejected(self, tmp_path: Path) -> None:
        q = tmp_path / "q.yaml"
        self._write_queue(
            q,
            "jobs:\n  - {name: vggt-co3dv2-pose}\n  - {name: vggt-co3dv2-pose}\n",
        )
        with pytest.raises(ValueError, match="duplicate"):
            queue_mod.load_queue(q)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            queue_mod.load_queue(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# run_queue behaviour (synthetic queue + reproductions)
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


class TestRunQueue:
    """Drive run_queue against synthetic reproductions in a tmp repro dir.

    ``repro_dir`` (imported fixture) redirects ``REPRODUCTIONS_DIR`` and the
    cache; ``registered_fakes`` registers the constant-AbsRel model + synthetic
    dataset. We point the queue file at that same tmp dir so ``_find_config``
    resolves the synthetic reproductions.
    """

    @pytest.fixture(autouse=True)
    def _point_queue_at_tmp(
        self, repro_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[None]:
        # _find_config inside queue.load_queue uses reproduce.REPRODUCTIONS_DIR,
        # already monkeypatched to repro_dir by the repro_dir fixture.
        self.repro_dir = repro_dir
        self.queue_path = repro_dir / "gpu_queue.yaml"
        monkeypatch.setattr(queue_mod, "QUEUE_PATH", self.queue_path)
        yield

    def _make_repro(self, name: str, *, target: float, value: float, tol: float = 0.05) -> None:
        # Per-job cache_dir: the fake model's default config_hash is keyed only
        # on name@version, so two jobs sharing model+dataset+sample_ids would
        # otherwise collide in one cache dir (the second cache-hits the first's
        # predictions). Real adapters fold their tunable kwargs into
        # config_hash; the fake doesn't, so isolate caches here.
        cache_dir = self.repro_dir / f"cache_{name.replace('-', '_')}"
        _write(
            self.repro_dir / f"{name.replace('-', '_')}.yaml",
            f"""
name: {name}
model: {{name: test-fixed-depth, kwargs: {{target_abs_rel: {target}}}}}
dataset: {{name: test-synthetic, kwargs: {{n_samples: 2}}}}
tasks: [mono_depth]
scale_alignment: none
device: cpu
cache_dir: {cache_dir}
paper_reference:
  primary_metric: abs_rel
  value: {value}
  tolerance_relative: {tol}
""",
        )

    def test_runs_pending_records_match_and_mismatch(self, registered_fakes: None) -> None:
        self._make_repro("good", target=0.10, value=0.10)
        self._make_repro("bad", target=0.20, value=0.10)
        _write(
            self.queue_path,
            """
jobs:
  - {name: good, status: pending, priority: 1}
  - {name: bad, status: pending, priority: 2}
""",
        )
        records = queue_mod.run_queue()
        by_name = {r.name: r for r in records}
        assert by_name["good"].outcome == "match"
        assert by_name["bad"].outcome == "mismatch"
        assert by_name["good"].observed == pytest.approx(0.10, abs=1e-6)

    def test_blocked_jobs_not_run(self, registered_fakes: None) -> None:
        self._make_repro("good", target=0.10, value=0.10)
        self._make_repro("nope", target=0.10, value=0.10)
        _write(
            self.queue_path,
            """
jobs:
  - {name: good, status: pending, priority: 1}
  - {name: nope, status: blocked, priority: 2, blocked_on: "test"}
""",
        )
        records = queue_mod.run_queue()
        assert [r.name for r in records] == ["good"]

    def test_only_filters_to_one_job(self, registered_fakes: None) -> None:
        self._make_repro("a", target=0.10, value=0.10)
        self._make_repro("b", target=0.10, value=0.10)
        _write(
            self.queue_path,
            """
jobs:
  - {name: a, status: pending, priority: 1}
  - {name: b, status: pending, priority: 2}
""",
        )
        records = queue_mod.run_queue(only="b")
        assert [r.name for r in records] == ["b"]

    def test_job_error_recorded_not_raised(self, registered_fakes: None) -> None:
        """A job that blows up (unknown dataset) is recorded as error; the
        queue keeps going and runs the next job."""
        self._make_repro("ok", target=0.10, value=0.10)
        _write(
            self.repro_dir / "broken.yaml",
            """
name: broken
model: {name: test-fixed-depth}
dataset: {name: no-such-dataset}
tasks: [mono_depth]
device: cpu
paper_reference: {primary_metric: abs_rel, value: 0.10}
""",
        )
        _write(
            self.queue_path,
            """
jobs:
  - {name: broken, status: pending, priority: 1}
  - {name: ok, status: pending, priority: 2}
""",
        )
        records = queue_mod.run_queue()
        by_name = {r.name: r for r in records}
        assert by_name["broken"].outcome == "error"
        assert "no-such-dataset" in by_name["broken"].error
        assert by_name["ok"].outcome == "match"

    def test_info_outcome_when_no_paper_value(self, registered_fakes: None) -> None:
        _write(
            self.repro_dir / "informational.yaml",
            """
name: informational
model: {name: test-fixed-depth}
dataset: {name: test-synthetic, kwargs: {n_samples: 2}}
tasks: [mono_depth]
scale_alignment: none
device: cpu
paper_reference: {primary_metric: abs_rel}
""",
        )
        _write(self.queue_path, "jobs:\n  - {name: informational, status: pending}\n")
        records = queue_mod.run_queue()
        assert records[0].outcome == "info"

    def test_output_summary_written(self, registered_fakes: None, tmp_path: Path) -> None:
        import json

        self._make_repro("good", target=0.10, value=0.10)
        _write(self.queue_path, "jobs:\n  - {name: good, status: pending}\n")
        out = tmp_path / "summary.json"
        queue_mod.run_queue(output=out)
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["records"][0]["name"] == "good"
        assert payload["records"][0]["outcome"] == "match"
