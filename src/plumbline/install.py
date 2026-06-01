"""Single source of truth for adapter install / dependency knowledge.

Every registered model adapter (see ``plumbline.models.registry``) has exactly
one :class:`InstallSpec` here. The :func:`spec_for`, :func:`install_hint`,
:func:`install_plan`, and :func:`check` helpers derive *all* downstream install
surface from these specs: the ``plumbline install`` / ``plumbline doctor`` CLI
commands, the adapters' own ``ImportError`` hints, the ``pyproject.toml``
optional-dependency extras, ``GPU_RUNBOOK.md``, and the GPU queue's ``extras:``
text are validated against (or generated from) this registry so the install
story can no longer drift.

Why this module exists: the GPU queue once claimed Depth-Anything-3 "ships in
the base install (transformers)", but the adapter actually needs
``pip install depth-anything-3`` — that drift wasted a GPU job. Encoding the
truth once, and validating everything else against it, is the fix.

Import discipline
-----------------
This module must import with **zero adapter dependencies** (no torch, no
transformers, no adapter modules). It is imported by adapter ``_load`` methods
for their ``ImportError`` hints, so importing any adapter back from here would
create a cycle. Keep it standard-library-only.

Adapter "kinds"
---------------
- ``base``  — installable from the published wheel; no extra step. ``check``
  passes iff ``probe_import`` imports.
- ``pypi``  — a published PyPI package (``pip install <name>``).
- ``git``   — a pip-installable VCS URL (``pip install git+https://...``).
- ``clone`` — a repo that must be ``git clone``d to a path named by an env var
  and added to ``sys.path`` by the adapter (not pip-installable). ``check``
  passes iff ``dest_env`` is set and points at an existing directory.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

__all__ = [
    "INSTALL_SPECS",
    "InstallSpec",
    "check",
    "install_hint",
    "install_plan",
    "spec_for",
]


@dataclass(frozen=True)
class InstallSpec:
    """Declarative install recipe for one adapter.

    Attributes
    ----------
    name:
        Registered model name (matches ``@register_model`` / ``MODEL_REGISTRY``).
    kind:
        One of ``"base" | "pypi" | "git" | "clone"`` (see module docstring).
    probe_import:
        Importable module name used to detect presence, or ``None`` when the
        adapter has no clean import probe (e.g. torch.hub-only, or a clone whose
        modules only resolve after ``sys.path`` munging).
    pip:
        Plain PyPI package specs for ``kind == "pypi"``.
    git:
        Pip-installable VCS URLs for ``kind == "git"`` (each goes to
        ``pip install <url>``).
    clone_url:
        Repo URL to ``git clone`` for ``kind == "clone"``.
    clone_recursive:
        Whether the clone needs ``--recursive`` (submodules).
    dest_env:
        Env var that points at the clone destination (clone kinds).
    default_dest:
        Default clone destination if ``dest_env`` is unset.
    extra_pip:
        Additional PyPI packages a clone needs (installed alongside the clone).
    requirements_txt:
        ``True`` if a clone installs its deps via its own ``requirements.txt``.
    extra_env:
        ``(var, note)`` pairs for *additional* env vars the adapter reads
        beyond ``dest_env`` (e.g. a checkpoint path, or a sibling repo root).
    weights:
        ``"hf-auto"`` (downloaded from the HF hub on first use) or
        ``"manual: <note>"`` (operator must place the checkpoint).
    pyproject_extra:
        Name of the matching ``[project.optional-dependencies]`` extra, if any.
    notes:
        Free-form operator notes (gotchas, optional speedups, ordering).
    """

    name: str
    kind: str
    probe_import: str | None = None
    pip: tuple[str, ...] = ()
    git: tuple[str, ...] = ()
    clone_url: str | None = None
    clone_recursive: bool = False
    dest_env: str | None = None
    default_dest: str | None = None
    extra_pip: tuple[str, ...] = ()
    requirements_txt: bool = False
    extra_env: tuple[tuple[str, str], ...] = ()
    weights: str = "hf-auto"
    pyproject_extra: str | None = None
    notes: str = ""

    def how_to(self) -> str:
        """One-line, human-facing how-to-install summary (for tables)."""
        if self.kind == "base":
            return "ships in the base install"
        if self.kind == "pypi":
            return f"plumbline install {self.name}  (pip: {' '.join(self.pip)})"
        if self.kind == "git":
            return f"plumbline install {self.name}  (pip+git)"
        # clone
        dest = self.dest_env or "<dest>"
        return f"plumbline install {self.name}  (git clone -> ${dest})"


# ---------------------------------------------------------------------------
# The registry — one entry per registered adapter. THIS IS THE SOURCE OF TRUTH.
# ---------------------------------------------------------------------------

_TRIMESH_STACK: tuple[str, ...] = ("roma", "scikit-learn", "trimesh")

INSTALL_SPECS: dict[str, InstallSpec] = {
    "metric3d-v2": InstallSpec(
        name="metric3d-v2",
        kind="base",
        probe_import=None,  # torch.hub.load("YvanYin/Metric3D") — no importable pkg
        weights="hf-auto",
        notes=(
            "Loaded via torch.hub.load('YvanYin/Metric3D', trust_repo=True). "
            "Its hubconf.py imports `mmcv.utils.Config` at module top, so the "
            "hub load fails with ModuleNotFoundError: No module named 'mmcv' "
            "unless mmcv is installed — and it wants the legacy 1.x API "
            "(`mmcv.utils`, moved to mmengine in 2.x), which has no prebuilt "
            "wheel for recent torch/CUDA and is painful to build. Treat "
            "metric3d-v2 as effectively blocked on torch>=2.x boxes until that "
            "is resolved. Gotcha (once it loads): a mismatched xformers wheel "
            "makes the dinov2 backbone raise NotImplementedError at forward "
            "time — uninstall xformers or match the wheel."
        ),
    ),
    "marigold": InstallSpec(
        name="marigold",
        kind="base",
        probe_import="diffusers",  # base dependency
        weights="hf-auto",
        notes="Uses diffusers (a base dependency); weights auto-download from HF.",
    ),
    "depth-anything-v2": InstallSpec(
        name="depth-anything-v2",
        kind="base",
        probe_import="transformers",  # importable via base deps; see notes on source=
        weights="hf-auto",
        extra_env=(
            (
                "DAV2_ROOT",
                "Required by the adapter's DEFAULT source='paper' path: a clone "
                "of https://github.com/DepthAnything/Depth-Anything-V2 "
                "(default /workspace/deps/depth-anything-v2), PLUS the clone's "
                "own deps — notably opencv-python (its dpt.py does `import cv2`); "
                "`uv pip install opencv-python`. The .pth weights auto-download "
                "from HF; only the model class comes from the clone.",
            ),
        ),
        notes=(
            "transformers (base dep) makes this import-OK, but the adapter "
            "DEFAULTS to source='paper' (the .pth checkpoints — the HF '-hf' "
            "re-exports score ~0.002 AbsRel lower and tip cells off-gate, so "
            "paper-match reproductions use 'paper'). That path needs a clone of "
            "https://github.com/DepthAnything/Depth-Anything-V2 at $DAV2_ROOT "
            "(default /workspace/deps/depth-anything-v2) AND its requirements "
            "(opencv-python — the repo's dpt.py imports cv2 at module top; "
            "without it the adapter raises a confusing 'needs the repo' error "
            "even when the clone is present). Without the paper backend every "
            "sample errors out and the run lands n_evaluated=0 / observed=nan. "
            "Pass source='hf' to use the no-clone transformers path instead."
        ),
    ),
    "depth-anything-3": InstallSpec(
        name="depth-anything-3",
        kind="pypi",
        probe_import="depth_anything_3",
        pip=("depth-anything-3",),
        weights="hf-auto",
    ),
    "moge": InstallSpec(
        name="moge",
        kind="git",
        probe_import="moge",
        git=("git+https://github.com/microsoft/MoGe.git",),
        weights="hf-auto",
    ),
    "vggt": InstallSpec(
        name="vggt",
        kind="git",
        probe_import="vggt",
        git=("git+https://github.com/facebookresearch/vggt",),
        weights="hf-auto",
        notes="Weights: facebook/VGGT-1B (HF auto).",
    ),
    "depth-pro": InstallSpec(
        name="depth-pro",
        kind="git",
        probe_import="depth_pro",
        git=("git+https://github.com/apple/ml-depth-pro.git",),
        weights=(
            "manual: depth_pro.pt is NOT on the HF hub — download it to "
            "~/.cache/plumbline/weights/depth-pro/depth_pro.pt"
        ),
    ),
    "mast3r": InstallSpec(
        name="mast3r",
        kind="clone",
        probe_import="mast3r",
        clone_url="https://github.com/naver/mast3r",
        clone_recursive=True,
        dest_env="MAST3R_ROOT",
        default_dest="$HOME/deps/mast3r",
        extra_pip=_TRIMESH_STACK,
        extra_env=(
            (
                "DUST3R_ROOT",
                "Path to the bundled dust3r submodule (default $MAST3R_ROOT/dust3r).",
            ),
        ),
        weights="hf-auto",
        notes=(
            "Optional curope CUDA build at $MAST3R_ROOT/dust3r/croco/models/curope "
            "for a speedup: `python setup.py build_ext --inplace` with "
            "CUDA_HOME=/usr/local/cuda-12.1. Only ONE dust3r-family model "
            "(mast3r/dust3r/monst3r) can be used per process — Python caches "
            "`import dust3r`."
        ),
    ),
    "dust3r": InstallSpec(
        name="dust3r",
        kind="clone",
        probe_import="dust3r",
        clone_url="https://github.com/naver/dust3r",
        clone_recursive=True,
        dest_env="DUST3R_ROOT",
        default_dest="$HOME/deps/dust3r",
        extra_pip=_TRIMESH_STACK,
        weights="hf-auto",
        notes=(
            "Weights: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt (HF auto). Only "
            "ONE dust3r-family model (mast3r/dust3r/monst3r) per process."
        ),
    ),
    "monst3r": InstallSpec(
        name="monst3r",
        kind="clone",
        probe_import=None,  # ships its own dust3r fork; resolves after sys.path munge
        clone_url="https://github.com/Junyi42/monst3r",
        clone_recursive=True,
        dest_env="MONST3R_ROOT",
        default_dest="$HOME/deps/monst3r",
        # MonST3R's dust3r fork imports `evo` at module load (dust3r/utils/
        # vo_eval.py), so it is required even for the mono-depth path — not
        # just trajectory metrics. roma/scikit-learn/trimesh are the shared
        # dust3r-family deps.
        extra_pip=(*_TRIMESH_STACK, "evo"),
        weights="hf-auto",
        notes=(
            "Ships its own dust3r fork — only ONE dust3r-family model "
            "(mast3r/dust3r/monst3r) can be used per process (Python caches "
            "`import dust3r`). The fork imports `evo` at load time, so it is a "
            "hard dependency (in extra_pip)."
        ),
    ),
    "cut3r": InstallSpec(
        name="cut3r",
        kind="clone",
        probe_import=None,  # resolves `src.dust3r...` only after sys.path munge
        clone_url="https://github.com/CUT3R/CUT3R",
        clone_recursive=True,
        dest_env="CUT3R_ROOT",
        default_dest="$HOME/deps/cut3r",
        requirements_txt=True,
        extra_env=(
            (
                "CUT3R_CKPT",
                "Path to the 512-DPT checkpoint (default $CUT3R_ROOT/src/cut3r_512_dpt_4_64.pth).",
            ),
        ),
        weights=(
            "manual: 512-DPT checkpoint cut3r_512_dpt_4_64.pth to $CUT3R_CKPT "
            "(default $CUT3R_ROOT/src/cut3r_512_dpt_4_64.pth). Public Google "
            "Drive id 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD (gdown)."
        ),
        notes=(
            "Deps install via the clone's requirements.txt (pulls omegaconf via "
            "hydra-core — needed to deserialise the checkpoint's embedded config; "
            "the adapter forces torch.load(weights_only=False) so torch>=2.6 "
            "doesn't reject that config). For GPU inference build the curope CUDA "
            "ext at $CUT3R_ROOT/src/croco/models/curope (`python setup.py "
            "build_ext --inplace`): unlike mast3r/dust3r (where curope is just a "
            "speedup) CUT3R's pure-torch RoPE fallback hits a device-side assert "
            "in apply_rope1d on some setups, so curope is effectively required."
        ),
    ),
    "geowizard": InstallSpec(
        name="geowizard",
        kind="clone",
        probe_import=None,  # upstream `models.geowizard_pipeline` resolves after sys.path munge
        clone_url="https://github.com/fuxiao0719/GeoWizard",
        clone_recursive=False,  # cloned with --depth 1
        dest_env="GEOWIZARD_ROOT",
        default_dest="$HOME/deps/geowizard",
        weights="hf-auto",
        pyproject_extra="geowizard",
        notes=(
            "Clone with --depth 1. Deps via the `geowizard` extra "
            "(xformers==0.0.29.post3), then "
            "`uv pip install --force-reinstall 'nvidia-cudnn-cu12==9.1.0.70'`. "
            "Weights: lemonaddie/Geowizard (HF auto)."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spec_for(name: str) -> InstallSpec:
    """Return the :class:`InstallSpec` for ``name`` (raises ``KeyError``)."""
    try:
        return INSTALL_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(INSTALL_SPECS))
        raise KeyError(f"no install spec for {name!r}; known: {known}") from exc


def install_hint(name: str) -> str:
    """Return a human ``ImportError``-style install hint for ``name``.

    Used by adapter ``_load`` methods so the message can't drift from the
    registry. Phrased as a sentence fragment that reads naturally after the
    adapter's own prefix (e.g. ``"DepthAnything3Adapter "`` + hint).
    """
    spec = spec_for(name)
    if spec.kind == "base":
        if spec.probe_import:
            return (
                f"needs `{spec.probe_import}`, which is a base dependency. "
                "Reinstall plumbline-bench."
            )
        return "ships in the base install; no extra step is required."
    if spec.kind == "pypi":
        pkg = spec.pip[0]
        mod = spec.probe_import or pkg
        return (
            f"needs the `{mod}` package. "
            f"Install with `plumbline install {name}` (or `uv pip install {pkg}`)."
        )
    if spec.kind == "git":
        url = spec.git[0]
        mod = spec.probe_import or name
        return (
            f"needs the `{mod}` package from {url}. "
            f"Install with `plumbline install {name}` (or `uv pip install {url}`)."
        )
    # clone
    dest = spec.dest_env or "<dest>"
    return (
        f"needs a clone of {spec.clone_url} at ${dest} "
        f"(default {spec.default_dest}). "
        f"Install with `plumbline install {name}`."
    )


def install_plan(name: str) -> list[str]:
    """Return ordered shell commands to install ``name``.

    - ``base``  → empty list (nothing to do).
    - ``pypi``  → one ``uv pip install <pkg>`` per package.
    - ``git``   → one ``uv pip install <url>`` per URL.
    - ``clone`` → ``git clone`` + extra-pip / requirements.txt + ``export`` lines
      for ``dest_env`` and any ``extra_env`` vars the operator must set.
    """
    spec = spec_for(name)
    if spec.kind == "base":
        return []
    if spec.kind == "pypi":
        return [f"uv pip install {pkg}" for pkg in spec.pip]
    if spec.kind == "git":
        return [f"uv pip install '{url}'" for url in spec.git]

    # clone
    dest = spec.default_dest or f"${spec.dest_env}"
    recursive = " --recursive" if spec.clone_recursive else ""
    plan = [f"git clone{recursive} {spec.clone_url} {dest}"]
    if spec.extra_pip:
        plan.append(f"uv pip install {' '.join(spec.extra_pip)}")
    if spec.requirements_txt:
        plan.append(f"uv pip install -r {dest}/requirements.txt")
    if spec.dest_env:
        plan.append(f"export {spec.dest_env}={dest}")
    for var, _note in spec.extra_env:
        plan.append(f"export {var}=...  # see notes")
    return plan


def _module_importable(module: str) -> bool:
    """True if ``module`` can be found without importing it (no side effects)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # find_spec raises if a *parent* package is missing or broken.
        return False


def check(name: str) -> tuple[bool, str]:
    """Report whether ``name`` looks installed. Returns ``(ok, detail)``.

    - For ``base`` / ``pypi`` / ``git`` adapters with a ``probe_import``: OK iff
      the module is importable (checked via ``find_spec`` — no heavy import).
    - For ``clone`` adapters (and anything without a clean probe): OK iff
      ``dest_env`` is set *and* points at an existing directory.
    """
    spec = spec_for(name)

    if spec.probe_import is not None:
        if _module_importable(spec.probe_import):
            return True, f"`{spec.probe_import}` importable"
        return False, f"`{spec.probe_import}` not importable — {install_hint(name)}"

    # No import probe: fall back to the clone-destination check.
    if spec.dest_env is not None:
        dest = os.environ.get(spec.dest_env)
        if not dest:
            return (
                False,
                f"${spec.dest_env} not set — {install_hint(name)}",
            )
        if not os.path.isdir(dest):
            return (
                False,
                f"${spec.dest_env}={dest} is not a directory — {install_hint(name)}",
            )
        return True, f"${spec.dest_env}={dest} present"

    # base adapter with no probe (metric3d-v2): nothing extra to install.
    return True, "ships in the base install"
