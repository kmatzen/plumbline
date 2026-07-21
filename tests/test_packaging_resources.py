"""Guard the runtime-data resolution that makes installed wheels usable.

``protocols/`` and ``reproductions/`` are read at runtime by
``load_protocol`` / ``_find_config``, but they live at the repo root — which
a pip-installed package does not have. The wheel force-includes both trees
under ``plumbline/`` and ``plumbline.paths`` falls back to them.

These tests pin both halves of that contract: the fallback logic itself, and
the build config that ships the data it falls back to.
"""

from __future__ import annotations

import re
from pathlib import Path

from plumbline.paths import PROTOCOLS_DIR, REPRODUCTIONS_DIR

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestResourceResolution:
    def test_dirs_resolve_in_the_checkout(self) -> None:
        assert PROTOCOLS_DIR.is_dir()
        assert REPRODUCTIONS_DIR.is_dir()
        assert list(PROTOCOLS_DIR.glob("*.yaml")), "no protocol presets found"

    def test_falls_back_to_packaged_copy_when_no_checkout(self, tmp_path: Path) -> None:
        """Simulate site-packages: no repo root, data bundled in the package."""
        from plumbline import paths

        fake_pkg = tmp_path / "site-packages" / "plumbline"
        (fake_pkg / "protocols").mkdir(parents=True)
        (fake_pkg / "protocols" / "demo.yaml").write_text("fixed: {}\n")

        original_pkg, original_root = paths._PACKAGE_DIR, paths.REPO_ROOT
        try:
            paths._PACKAGE_DIR = fake_pkg
            paths.REPO_ROOT = fake_pkg.parent.parent  # has no protocols/ dir
            assert paths._resource_dir("protocols") == fake_pkg / "protocols"
        finally:
            paths._PACKAGE_DIR, paths.REPO_ROOT = original_pkg, original_root

    def test_falls_back_to_repo_root_path_when_neither_exists(self, tmp_path: Path) -> None:
        """Neither the checkout nor the packaged copy is present: still
        name the developer-facing repo-root path (so error messages stay
        legible) rather than raising here."""
        from plumbline import paths

        fake_pkg = tmp_path / "site-packages" / "plumbline"
        fake_pkg.mkdir(parents=True)
        fake_root = tmp_path / "no-such-repo"

        original_pkg, original_root = paths._PACKAGE_DIR, paths.REPO_ROOT
        try:
            paths._PACKAGE_DIR = fake_pkg
            paths.REPO_ROOT = fake_root
            assert paths._resource_dir("protocols") == fake_root / "protocols"
        finally:
            paths._PACKAGE_DIR, paths.REPO_ROOT = original_pkg, original_root

    def test_prefers_checkout_over_packaged_copy(self, tmp_path: Path) -> None:
        """An editable/in-tree run must use the checkout, not a stale bundle."""
        from plumbline import paths

        root = tmp_path / "repo"
        pkg = root / "src" / "plumbline"
        (root / "protocols").mkdir(parents=True)
        (pkg / "protocols").mkdir(parents=True)

        original_pkg, original_root = paths._PACKAGE_DIR, paths.REPO_ROOT
        try:
            paths._PACKAGE_DIR, paths.REPO_ROOT = pkg, root
            assert paths._resource_dir("protocols") == root / "protocols"
        finally:
            paths._PACKAGE_DIR, paths.REPO_ROOT = original_pkg, original_root


class TestBuildConfigShipsRuntimeData:
    """Without these, an installed wheel raises FileNotFoundError per cell."""

    def test_wheel_force_includes_runtime_data(self) -> None:
        text = _PYPROJECT.read_text()
        section = text.split("[tool.hatch.build.targets.wheel.force-include]")[1]
        section = section.split("\n[")[0]
        assert re.search(r'"protocols"\s*=\s*"plumbline/protocols"', section)
        assert re.search(r'"reproductions"\s*=\s*"plumbline/reproductions"', section)

    def test_sdist_includes_runtime_data(self) -> None:
        text = _PYPROJECT.read_text()
        section = text.split("[tool.hatch.build.targets.sdist]")[1].split("\n[")[0]
        assert '"/protocols"' in section
        assert '"/reproductions"' in section
