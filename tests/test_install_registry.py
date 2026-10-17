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
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import plumbline.install as install_module
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

_VENDORED_BACKEND_IMPORTS = {
    "depth_anything_3": (
        "_ensure_da3_on_path",
        "from depth_anything_3.api import DepthAnything3",
    ),
    "moge": ("_ensure_moge_on_path", "from moge.model.v2 import MoGeModel"),
    "pi3": ("_ensure_pi3_on_path", "from pi3.models.pi3x import Pi3X"),
    "streamvggt": (
        "_ensure_streamvggt_on_path",
        "from streamvggt.models.streamvggt import StreamVGGT",
    ),
    "unik3d": ("_ensure_unik3d_on_path", "from unik3d.models import UniK3D"),
    "vda": (
        "_ensure_vda_on_path",
        "from video_depth_anything.video_depth import VideoDepthAnything",
    ),
}


def test_registry_is_nonempty_and_has_18_entries() -> None:
    # Guards against a registry that silently emptied (which would make the
    # parametrized cases vacuously pass) and pins the documented count.
    assert len(INSTALL_SPECS) == 18, sorted(INSTALL_SPECS)


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


def test_check_requires_every_declared_runtime_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """One installed dependency must not make a partially installed adapter OK."""
    available = {"einops", "omegaconf", "safetensors", "kornia", "roma"}
    monkeypatch.setattr(install_module, "_module_importable", available.__contains__)

    ok, detail = check("dage")

    assert not ok
    assert "segmentation_models_pytorch" in detail


def test_vendored_adapter_probes_external_deps_not_private_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vendored package roots are activated lazily by adapters, not site-packages."""
    available = {"addict", "omegaconf", "cv2"}
    monkeypatch.setattr(install_module, "_module_importable", available.__contains__)

    ok, detail = check("depth-anything-3")

    assert ok, detail


def test_doctor_rejects_missing_manual_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(install_module, "_module_importable", lambda _module: True)
    monkeypatch.setenv("HOME", str(tmp_path))

    ok, detail = check("depth-pro")

    assert not ok
    assert "depth_pro.pt" in detail
    assert "missing required files" in detail


def test_doctor_accepts_manual_checkpoint_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "cut3r.pth"
    checkpoint.touch()
    monkeypatch.setattr(install_module, "_module_importable", lambda _module: True)
    monkeypatch.setenv("CUT3R_CKPT", str(checkpoint))

    ok, detail = check("cut3r")

    assert ok, detail
    assert "required files present" in detail


def test_every_manual_weight_spec_declares_a_required_file() -> None:
    for name, spec in INSTALL_SPECS.items():
        if spec.weights.startswith("manual:"):
            assert spec.required_files, f"{name}: manual weights are invisible to doctor"


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
            # Only executable pip steps: env vars are optional overrides or
            # manual-artifact guidance and cannot persist from a subprocess.
            assert all(s.startswith("uv pip install") for s in plan), plan
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


def test_queue_does_not_require_optional_vendored_root_overrides() -> None:
    jobs = _load_queue_jobs()
    for job in jobs:
        required = set(job.get("requires_env", ()))
        if job["name"].startswith("cut3r-"):
            assert "CUT3R_ROOT" not in required, job["name"]
        if job["name"].startswith("mast3r-"):
            assert "MAST3R_ROOT" not in required, job["name"]


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


def test_moge_is_vendored_and_needs_no_external_install() -> None:
    """MoGe v1/v2 ships in-tree; upstream HEAD is incompatible MoGe 3."""
    spec = spec_for("moge")
    assert spec.kind == "vendored"
    assert spec.pip == ()
    assert spec.git == ()
    assert install_plan("moge") == []


def test_base_complete_vendored_installs_are_true_noops() -> None:
    assert install_plan("depth-anything-3") == []
    assert install_plan("pi3") == []
    assert install_plan("streamvggt") == []


@pytest.mark.parametrize("module_name", sorted(_VENDORED_BACKEND_IMPORTS))
def test_base_complete_vendored_backend_imports(module_name: str) -> None:
    """Import the real backend, not merely its dependency-light wrapper.

    Each backend gets a fresh interpreter because upstream packages mutate
    ``sys.path`` and several use generic top-level module names. Importing the
    canonical model class exercises the bundled tree and its transitive runtime
    dependencies without downloading weights or requiring a GPU.
    """
    helper, backend_import = _VENDORED_BACKEND_IMPORTS[module_name]
    source = f"from plumbline.models.{module_name} import {helper}; {helper}(); {backend_import}"
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_install_plans_never_include_exports() -> None:
    for name, spec in INSTALL_SPECS.items():
        if spec.kind == "vendored":
            assert not any(step.startswith("export ") for step in install_plan(name))


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
