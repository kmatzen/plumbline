"""Static gate for the adapter install registry (``plumbline.install``).

``src/plumbline/install.py`` is the single source of truth for how every
adapter is installed. These tests keep it honest without a GPU, model weights,
or any adapter dependency (no torch / transformers import):

1. The registry and ``MODEL_REGISTRY`` are 1:1 — no adapter without an install
   spec, no spec without an adapter.
2. Each spec is internally consistent for its ``kind`` (pypi⇒pip set, git⇒git
   set, clone⇒clone_url + dest_env set).
3. Regression for the original bug: the ``da3-co3dv2-pose`` GPU-queue job no
   longer claims DA3 "ships in the base install" — it needs a pip package.
4. ``install_plan`` / ``install_hint`` / ``check`` run cleanly for all 13.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plumbline.install import (
    INSTALL_SPECS,
    check,
    install_hint,
    install_plan,
    spec_for,
)

_ALL_NAMES = sorted(INSTALL_SPECS)
_VALID_KINDS = {"base", "pypi", "git", "clone"}

_GPU_QUEUE_YAML = Path(__file__).resolve().parent.parent / "reproductions" / "gpu_queue.yaml"
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_registry_is_nonempty_and_has_13_entries() -> None:
    # Guards against a registry that silently emptied (which would make the
    # parametrized cases vacuously pass) and pins the documented count.
    assert len(INSTALL_SPECS) == 13, sorted(INSTALL_SPECS)


def test_registry_matches_model_registry_exactly() -> None:
    # Importing the registries triggers built-in adapter discovery, but those
    # imports are soft (missing optional deps are swallowed), so do it here.
    from plumbline._discover import register_builtin_adapters
    from plumbline.models.registry import MODEL_REGISTRY

    register_builtin_adapters()
    model_names = set(MODEL_REGISTRY)
    spec_names = set(INSTALL_SPECS)
    assert spec_names == model_names, {
        "specs_without_adapter": sorted(spec_names - model_names),
        "adapters_without_spec": sorted(model_names - spec_names),
    }


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_spec_name_matches_key_and_kind_is_valid(name: str) -> None:
    spec = spec_for(name)
    assert spec.name == name
    assert spec.kind in _VALID_KINDS, spec.kind


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_spec_is_internally_consistent_for_its_kind(name: str) -> None:
    spec = spec_for(name)
    if spec.kind == "pypi":
        assert spec.pip, f"{name}: pypi kind must set `pip`"
        assert not spec.git and spec.clone_url is None, f"{name}: pypi must not set git/clone"
    elif spec.kind == "git":
        assert spec.git, f"{name}: git kind must set `git`"
        assert not spec.pip and spec.clone_url is None, f"{name}: git must not set pip/clone"
    elif spec.kind == "clone":
        assert spec.clone_url is not None, f"{name}: clone kind must set `clone_url`"
        assert spec.dest_env, f"{name}: clone kind must set `dest_env`"
        assert not spec.pip and not spec.git, f"{name}: clone must not set pip/git"
    else:  # base
        assert not spec.pip and not spec.git and spec.clone_url is None, (
            f"{name}: base kind must not set pip/git/clone"
        )


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_pyproject_extra_exists_in_pyproject(name: str) -> None:
    # Anti-drift: a spec's pyproject_extra must name a REAL extra in
    # pyproject.toml. Adapter deps install via `plumbline install` (git URLs
    # can't be published extras; heavy trees pollute uv.lock), so in practice
    # only geowizard declares one (its lone PyPI dep, xformers).
    spec = spec_for(name)
    if spec.pyproject_extra is None:
        return
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    assert spec.pyproject_extra in extras, (
        f"{name}: registry pyproject_extra={spec.pyproject_extra!r} is not a real "
        f"extra in pyproject.toml [project.optional-dependencies] ({sorted(extras)})."
    )


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_helpers_run_without_error(name: str) -> None:
    plan = install_plan(name)
    assert isinstance(plan, list)
    assert all(isinstance(s, str) for s in plan)

    hint = install_hint(name)
    assert isinstance(hint, str) and hint

    ok, detail = check(name)
    assert isinstance(ok, bool)
    assert isinstance(detail, str) and detail


def test_install_plan_shape_per_kind() -> None:
    for name in _ALL_NAMES:
        spec = spec_for(name)
        plan = install_plan(name)
        if spec.kind == "base":
            assert plan == [], f"{name}: base plan must be empty"
        elif spec.kind in ("pypi", "git"):
            assert plan, f"{name}: {spec.kind} plan must have steps"
            assert all(s.startswith("uv pip install") for s in plan), plan
        else:  # clone
            assert plan[0].startswith("git clone"), plan
            assert any(s.startswith("export") for s in plan), plan


# ---------------------------------------------------------------------------
# Regression for the original bug: the GPU queue once lied that DA3 ships in
# the base install. The fix re-points it at the pip package.
# ---------------------------------------------------------------------------


def _load_queue_jobs() -> list[dict]:
    data = yaml.safe_load(_GPU_QUEUE_YAML.read_text(encoding="utf-8"))
    return data["jobs"]


def test_gpu_queue_yaml_parses() -> None:
    jobs = _load_queue_jobs()
    assert jobs, "gpu_queue.yaml has no jobs"


def test_da3_job_no_longer_claims_base_install() -> None:
    jobs = _load_queue_jobs()
    da3 = next((j for j in jobs if j["name"] == "da3-co3dv2-pose"), None)
    assert da3 is not None, "da3-co3dv2-pose job missing from gpu_queue.yaml"
    extras = da3.get("extras", "")
    assert "base install" in extras, "expected the extras line to mention 'base install' (negated)"
    # The lie we fixed: it must NOT claim DA3 *ships in* the base install.
    assert "base install (transformers)" not in extras, extras
    assert "ships in the base install" not in extras, extras
    # And it must point operators at the real install path.
    assert "depth-anything-3" in extras, extras


def test_da3_extras_matches_registry_truth() -> None:
    # The fixed extras text must agree with the registry: DA3 is a pypi adapter.
    spec = spec_for("depth-anything-3")
    assert spec.kind == "pypi"
    assert spec.pip == ("depth-anything-3",)
