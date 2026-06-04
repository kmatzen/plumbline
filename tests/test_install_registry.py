"""Static gate for the adapter install registry (``plumbline.install``).

``src/plumbline/install.py`` is the single source of truth for how every
adapter is installed. These tests keep it honest without a GPU, model weights,
or any adapter dependency (no torch / transformers import):

1. The registry and ``MODEL_REGISTRY`` are 1:1 — no adapter without an install
   spec, no spec without an adapter.
2. Each spec is internally consistent for its ``kind`` (pypi⇒pip set, git⇒git
   set, clone⇒clone_url + dest_env set, vendored⇒vendorable + tree on disk + no
   clone; its deps live in ``pip`` or — for a light slice like DA3 — base).
3. The ``da3-co3dv2-pose`` GPU-queue job tells the truth about DA3's install
   path (now vendored with deps in base; formerly a pip package).
4. ``install_plan`` / ``install_hint`` / ``check`` run cleanly for all adapters.
"""

from __future__ import annotations

import re
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
_VALID_KINDS = {"base", "pypi", "git", "clone", "vendored"}

_GPU_QUEUE_YAML = Path(__file__).resolve().parent.parent / "reproductions" / "gpu_queue.yaml"
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
_MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "plumbline" / "models"
_HINT_CALL = re.compile(r"""install_hint\(\s*['"]([^'"]+)['"]\s*\)""")


def test_registry_is_nonempty_and_has_17_entries() -> None:
    # Guards against a registry that silently emptied (which would make the
    # parametrized cases vacuously pass) and pins the documented count.
    assert len(INSTALL_SPECS) == 17, sorted(INSTALL_SPECS)


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
    elif spec.kind == "vendored":
        # Code ships under _vendor/<name> (the env-var override lives in
        # extra_env, not dest_env). Runtime deps install via `pip` OR — for a
        # slice light enough to live in base (DA3: addict/omegaconf/opencv) —
        # via base pyproject, in which case `pip` is empty. No clone/git, must
        # be vendorable, and the tree must actually exist on disk.
        assert spec.clone_url is None and not spec.git, f"{name}: vendored must not set clone/git"
        assert spec.vendorable, f"{name}: vendored kind must be vendorable"
        vdir = _PYPROJECT.parent / "src" / "plumbline" / "_vendor" / name.replace("-", "_")
        assert vdir.is_dir(), (
            f"{name}: vendored kind but {vdir.relative_to(_PYPROJECT.parent)} is missing"
        )
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
        elif spec.kind == "vendored":
            # pip-install lines (+ optional export lines for checkpoint paths);
            # no clone step (code is bundled under _vendor/).
            assert plan, f"{name}: vendored plan must have steps"
            assert all(s.startswith(("uv pip install", "export")) for s in plan), plan
            assert not any(s.startswith("git clone") for s in plan), plan
        else:  # clone
            assert plan[0].startswith("git clone"), plan
            assert any(s.startswith("export") for s in plan), plan


# ---------------------------------------------------------------------------
# DA3 install story. Originally a bug (queue lied DA3 ships in base → fixed by
# pointing at the pip package). As of the 2026-06-04 vendor, DA3 *is* base-
# usable: the mono-depth subset is vendored under _vendor/depth_anything_3 and
# its light deps live in base pyproject, so no install step is needed.
# ---------------------------------------------------------------------------


def _load_queue_jobs() -> list[dict]:
    data = yaml.safe_load(_GPU_QUEUE_YAML.read_text(encoding="utf-8"))
    return data["jobs"]


def test_gpu_queue_yaml_parses() -> None:
    jobs = _load_queue_jobs()
    assert jobs, "gpu_queue.yaml has no jobs"


def test_da3_job_reflects_vendored_in_base() -> None:
    jobs = _load_queue_jobs()
    da3 = next((j for j in jobs if j["name"] == "da3-co3dv2-pose"), None)
    assert da3 is not None, "da3-co3dv2-pose job missing from gpu_queue.yaml"
    extras = da3.get("extras", "")
    # New truth: vendored, deps in base, no pip/install step.
    assert "vendored" in extras.lower(), extras
    assert "base install" in extras, extras
    assert "plumbline install depth-anything-3" not in extras, extras
    assert "pip install depth-anything-3" not in extras, extras


# ---------------------------------------------------------------------------
# Adapter ImportError messages derive from the registry via install_hint(); a
# typo there would make the error path itself raise KeyError. Every name passed
# to install_hint() in the adapter modules must be a real registry key.
# ---------------------------------------------------------------------------


def test_adapter_install_hint_args_are_valid_registry_names() -> None:
    calls: list[tuple[str, str]] = []
    for path in sorted(_MODELS_DIR.glob("*.py")):
        for m in _HINT_CALL.finditer(path.read_text(encoding="utf-8")):
            calls.append((path.name, m.group(1)))
    assert calls, "no install_hint(...) calls found in adapter modules — wiring removed?"
    for fname, name in calls:
        assert name in INSTALL_SPECS, (
            f"{fname}: install_hint({name!r}) names an unknown adapter; "
            f"valid: {sorted(INSTALL_SPECS)}"
        )


def test_da3_extras_matches_registry_truth() -> None:
    # DA3 is now VENDORED (mono-depth subset under _vendor/depth_anything_3) with
    # its light deps (addict/omegaconf/opencv-python) declared in base pyproject,
    # so there is nothing to pip-install — the registry must reflect that.
    spec = spec_for("depth-anything-3")
    assert spec.kind == "vendored"
    assert spec.pip == ()


def test_every_spec_records_a_license() -> None:
    # The vendoring policy ("license permitting") is only enforceable if every
    # model's upstream license is recorded. vendorable is gated on a non-GPL,
    # non-unlicensed, reviewed license (NonCommercial allowed).
    for name, spec in INSTALL_SPECS.items():
        assert spec.license, f"{name}: no upstream license recorded in _LICENSE_INFO"


def test_unlicensed_and_custom_are_not_vendorable() -> None:
    # GeoWizard has no LICENSE (all rights reserved) and must never be vendored;
    # bespoke licenses (VGGT/Apple) need a human clause-read first.
    assert INSTALL_SPECS["geowizard"].vendorable is False
    assert INSTALL_SPECS["vggt"].vendorable is False
    assert INSTALL_SPECS["depth-pro"].vendorable is False
