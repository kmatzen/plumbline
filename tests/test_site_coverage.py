"""Guard the landing page's coverage lists against drift.

The site (``site/index.html``) advertises the supported models + datasets.
When an adapter is added/removed the page silently goes stale — exactly how it
ended up claiming "9 models / 8 datasets" after π³ (``pi3``) and the
GSO/iBims-1/7-Scenes loaders had landed. These tests assert:

- every model/dataset the page lists is actually registered, and
- the displayed count matches the number of listed items, and
- the model list is *complete* (every registered model adapter appears).

Datasets aren't required to be exhaustive (the registry also holds
eval-protocol variants like ``kitti-moge-eval`` that the page intentionally
folds into their base dataset), but every name shown must resolve.
"""

from __future__ import annotations

import re

import pytest

from plumbline._discover import register_builtin_adapters
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.paths import REPO_ROOT

SITE_HTML = REPO_ROOT / "site" / "index.html"
EXPLORE_HTML = REPO_ROOT / "site" / "explore.html"
REPRODUCTIONS_MD = REPO_ROOT / "REPRODUCTIONS.md"
README_MD = REPO_ROOT / "README.md"


def _parse_coverage(kind: str) -> tuple[int, list[str]]:
    """Return (displayed_count, listed_items) for the ``models`` or
    ``datasets`` column in the supported/ section.

    Matches the ``<h3>{kind}<span class="cnt">N</span></h3> ... </article>``
    block and pulls the count and each ``<li>`` entry from it.
    """
    html = SITE_HTML.read_text(encoding="utf-8")
    block = re.search(
        rf'<h3>{kind}<span class="cnt">(\d+)</span></h3>(.*?)</article>',
        html,
        re.DOTALL,
    )
    assert block, f"could not find the {kind!r} coverage column in {SITE_HTML}"
    count = int(block.group(1))
    items = re.findall(r"<li>([^<]+)</li>", block.group(2))
    return count, [i.strip() for i in items]


@pytest.fixture(scope="module", autouse=True)
def _adapters() -> None:
    register_builtin_adapters()


class TestSiteCoverage:
    def test_models_count_matches_list_length(self) -> None:
        count, items = _parse_coverage("models")
        assert count == len(items), f"site claims {count} models but lists {len(items)}"

    def test_every_listed_model_is_registered(self) -> None:
        _, items = _parse_coverage("models")
        unknown = [m for m in items if m not in MODEL_REGISTRY]
        assert not unknown, f"site lists unregistered models: {unknown}"

    def test_model_list_is_complete(self) -> None:
        """Every registered model adapter must appear on the page."""
        _, items = _parse_coverage("models")
        missing = sorted(set(MODEL_REGISTRY) - set(items))
        assert not missing, (
            f"site is missing registered models: {missing} "
            f"(update site/index.html supported/ section)"
        )

    def test_datasets_count_matches_list_length(self) -> None:
        count, items = _parse_coverage("datasets")
        assert count == len(items), f"site claims {count} datasets but lists {len(items)}"

    def test_every_listed_dataset_is_registered(self) -> None:
        _, items = _parse_coverage("datasets")
        unknown = [d for d in items if d not in DATASET_REGISTRY]
        assert not unknown, f"site lists unregistered datasets: {unknown}"


# --- Cross-surface verified-count consistency (added 2026-05-28) ---
# The "verified paper-match" count must agree across every surface that
# advertises it: the site stat, the site's listed cells, REPRODUCTIONS.md's
# breakdown total, and the README headline. Caught the 24/23/22 drift fixed
# 2026-05-28 (two over-claimed ✅ cells downgraded + a generation-stale README).


def _site_verified_count() -> int:
    html = SITE_HTML.read_text(encoding="utf-8")
    m = re.search(
        r'<span class="stat-n">(\d+)</span><span class="stat-label">verified results',
        html,
    )
    assert m, "could not find the 'verified results' stat on the site"
    return int(m.group(1))


def _explore_verified_cells() -> list[dict]:
    """The verified-cell data backing the interactive deviation field on
    site/explore.html (moved off the landing page 2026-06-04)."""
    import json

    html = EXPLORE_HTML.read_text(encoding="utf-8")
    block = re.search(
        r'<script type="application/json" id="verified-cells">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert block, "could not find the verified-cells JSON in site/explore.html"
    return json.loads(block.group(1))


def _site_verified_list_count() -> int:
    return len(_explore_verified_cells())


def _reproductions_verified_total() -> int:
    md = REPRODUCTIONS_MD.read_text(encoding="utf-8")
    m = re.search(r"=\s*(\d+)\s+total\*\*", md)
    assert m, "could not find the '= N total' verified breakdown in REPRODUCTIONS.md"
    return int(m.group(1))


class TestVerifiedCountConsistency:
    """The 'verified paper-match' count must agree across all surfaces."""

    def test_site_stat_matches_listed_cells(self) -> None:
        stat, listed = _site_verified_count(), _site_verified_list_count()
        assert stat == listed, (
            f"site 'verified results' stat ({stat}, site/index.html) != number of "
            f"verified-cell entries ({listed}, site/explore.html #verified-cells)"
        )

    def test_site_matches_reproductions_total(self) -> None:
        site, repro = _site_verified_count(), _reproductions_verified_total()
        assert site == repro, (
            f"site verified count ({site}) != REPRODUCTIONS.md '= N total' ({repro})"
        )

    def test_readme_counts_match_reproductions_total(self) -> None:
        readme = README_MD.read_text(encoding="utf-8")
        repro = _reproductions_verified_total()
        counts = {int(n) for n in re.findall(r"(\d+) paper-match", readme)}
        counts |= {int(n) for n in re.findall(r"(\d+)-cell matrix", readme)}
        assert counts, "no 'N paper-match' / 'N-cell matrix' count found in README"
        bad = sorted(c for c in counts if c != repro)
        assert not bad, (
            f"README paper-match counts {bad} disagree with REPRODUCTIONS.md total ({repro})"
        )
