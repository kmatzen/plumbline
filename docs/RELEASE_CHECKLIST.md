# Release checklist — publishing `plumbline-bench` to PyPI

Status as of 2026-06-02: `v0.1.0` is **tagged but unpublished**. The release
*plumbing* is done (`.github/workflows/wheels.yml`: trusted-publishing OIDC,
`pypi` environment, fires on `v*` tags **and** `workflow_dispatch`). What
remains is two decisions, a packaging-honesty fix, and the GitHub/PyPI infra
that only the repo owner can set up.

Legend: **[owner]** = needs your GitHub/PyPI account · **[claude]** = I can do it
in-repo · **[joint]** = decide together, I implement.

---

## Tier 0 — decisions that block everything

- [ ] **[joint] Version.** The `v0.1.0` tag points at `b82d421` (2026-05-31),
      which is *before* any vendoring — no `_vendor/` exists at that commit, and
      ~15 PRs of adapters/vendoring landed after. Do **not** publish from the old
      tag. Options:
      - **Recommended:** cut **`v0.2.0`** from current `main` (vendoring is a
        substantial, user-visible change → minor bump). Move the CHANGELOG
        `[Unreleased]` block under a new `## [0.2.0]` heading.
      - Or delete + re-create `v0.1.0` (rewrites a public tag — avoid).

- [ ] **[joint] NC code in the distributed wheel.** `packages = ["src/plumbline"]`
      bundles all of `src/plumbline/_vendor/**` — i.e. the CC-BY-NC[-SA] DAGE /
      CUT3R / DUSt3R / MASt3R / MonST3R source — into the published wheel. You've
      said NC is acceptable; the remaining requirement is **honest metadata** so
      the PyPI page doesn't read as pure Apache-2.0. Pick one:
      - **Ship + relabel (recommended, matches the vendoring intent):** keep
        `_vendor` in the wheel; add an explicit "contains NonCommercial
        components — the combined distribution is usable for non-commercial
        purposes only" notice to the PyPI long description and a `License ::
        Other/Proprietary License`-style caveat. See Tier 1.
      - **Exclude from wheel:** add a hatch wheel `exclude` for `_vendor`, and
        fetch/clone the model code at install time. Defeats the point of
        vendoring (one-install adapters) — not recommended.

---

## Tier 1 — packaging correctness (in-repo) **[claude]**

Do these once Tier 0 is decided; all are code/doc changes I can make + verify
locally.

- [ ] **License honesty** (if shipping NC, per Tier 0): banner in `README.md`
      ("bundles NonCommercial model code; see THIRD_PARTY_NOTICES.md — the wheel
      is non-commercial as a whole"), and adjust the PyPI `classifiers` /
      `license` note accordingly. There is no OSI "NC" classifier, so this is a
      prose + `Private :: Do Not Upload`-free note, not a metadata enum.
- [ ] **Bundle the notices with the artifact.** `THIRD_PARTY_NOTICES.md` lives at
      repo root, so it is **not** in the wheel (`packages = src/plumbline` only).
      For an NC-bundling wheel the notices + top-level `LICENSE` must travel with
      it — add a hatch `force-include` (and add `THIRD_PARTY_NOTICES.md` +
      `CHANGELOG.md` to the sdist `include` list, currently README+LICENSE only).
      The per-vendor `_vendor/<m>/LICENSE` files already travel inside `_vendor`.
- [ ] **Lint + format + tests pass locally** (the `wheels.yml` `test` job gates
      the publish on all three, so a failure here = no publish): run
      `uv run ruff format --check src tests && uv run ruff check src tests && uv run pytest -q`.
      NOTE `ruff check` (lint) and `ruff format --check` (formatting) are
      **separate** — v0.2.1's first tag failed the format check because only
      lint was run locally. `pre-commit` covers both but only if installed as a
      git hook (`pre-commit install`); this line is the belt-and-braces.
- [ ] **Build + inspect locally:** `uv build`, then unzip the wheel and confirm
      (a) `_vendor/**` present (or absent, per decision), (b) curope `.cu/.cpp/.h`
      source included, (c) **no** `*.so` / `build/` / model weights, (d) wheel
      size is sane (tens of MB, not GB).
- [ ] **`twine check dist/*`** passes (the workflow runs it too — catch it early).
- [ ] **Version + CHANGELOG** finalized: bump `pyproject.toml` `version`, close the
      CHANGELOG section, fix the `[Unreleased]`/compare links.
- [ ] **README install story** reflects the vendored reality (no clones for the
      dust3r-lineage / dage; `plumbline install <m>` = pip deps only).

---

## Tier 2 — GitHub + PyPI infrastructure **[owner]**

- [ ] **Make the repo public** (Settings → General → Danger Zone). Trusted
      publishing works on private repos, but the project is meant to be OSS.
- [ ] **Create the `pypi` GitHub environment** (Settings → Environments). The
      workflow references `environment: pypi`; create it explicitly and
      (recommended) protect it: restrict deployments to `v*` tags, optionally a
      required reviewer. This is your manual gate on every publish.
- [ ] **Register the PyPI trusted publisher.** `plumbline-bench` doesn't exist on
      PyPI yet, so use a **pending publisher**: PyPI → Account → Publishing → add
      with Owner `kmatzen`, Repo `plumbline`, Workflow `wheels.yml`, Environment
      `pypi`. (Name must be exactly `plumbline-bench`.)
- [ ] **(Recommended) TestPyPI dry-run** to validate the OIDC handshake before
      the real publish. The throwaway workflow `.github/workflows/testpypi.yml`
      already exists — it is **manual-only** (so it can't race `wheels.yml`) and
      stamps a unique `0.2.0.devN` version per run (so it's re-runnable). To use:
      1. On **test.pypi.org** → Account → Publishing → add a *pending* publisher:
         Project `plumbline-bench`, Owner `kmatzen`, Repo `plumbline`, Workflow
         `testpypi.yml`, Environment `testpypi`.
      2. Create a **`testpypi`** GitHub environment (Settings → Environments).
      3. Actions → "testpypi-dryrun" → **Run workflow**. Confirm build + OIDC
         upload succeed; optionally `pip install -i https://test.pypi.org/simple/
         --extra-index-url https://pypi.org/simple/ plumbline-bench` in a clean
         venv (the extra-index lets the real torch/HF deps resolve from PyPI).
      4. **Delete `.github/workflows/testpypi.yml`** once the handshake is proven.

---

## Tier 3 — cut the release **[owner triggers · claude preps]**

- [ ] **CI green on `main`.** Note: the repo's Actions quota has been exhausted in
      the past — confirm it has reset (or top it up) before relying on the
      tag-triggered run.
- [ ] **Tag + push:** `git tag v0.2.0 && git push origin v0.2.0` → fires
      `wheels.yml` (test → build → publish). Or run it from the Actions tab via
      `workflow_dispatch` (build artifacts only; publish still needs the tag/env).
- [ ] **Smoke-test the published package** in a clean venv: `uv pip install
      plumbline-bench`, `python -c "import plumbline"`, `plumbline doctor` (CPU,
      no weights). Confirm a vendored adapter resolves (`plumbline install dust3r`
      plan prints the pip deps, import path finds `_vendor/dust3r`).
- [ ] **Create the GitHub Release** for the tag with notes from the CHANGELOG.

---

## Tier 4 — post-publish **[owner / joint]**

- [ ] **Site:** drop the Basic Auth and make `plumbline-bench.org` public
      (`scripts/deploy_site_preview.py --no-auth`), per the site-deploy notes.
- [ ] **Announce** + update README badges (PyPI version, CI).

---

### One-line summary of the gating path
Decide **version (→ v0.2.0)** and **NC-in-wheel (→ ship + relabel)** → I fix the
packaging metadata + bundle the notices → you make the repo public, create the
`pypi` env, and register the PyPI pending-publisher → tag `v0.2.0` → it builds
and publishes itself.
