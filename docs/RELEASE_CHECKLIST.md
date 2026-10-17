# Release checklist — `plumbline-bench`

Status verified 2026-10-16: versions 0.2.0 and 0.2.1 are published on PyPI.
The repository is public, the `pypi` GitHub environment exists, and the trusted
publishing workflow completed successfully for both releases. The website is
deployed behind its existing Basic Auth gate; changing that audience remains a
separate owner decision.

This checklist describes the next release. Do not reuse old version numbers.

## Prepare the release

- [ ] Choose the next version from the changes since 0.2.1.
- [ ] Update `pyproject.toml`, `src/plumbline/_version.py`, `CITATION.cff`, and
      the CHANGELOG release heading together. `tests/test_version_sync.py`
      checks the first three version values.
- [ ] Confirm the README describes the current research scope, install process,
      bundled non-commercial components, and exact-reproduction constraints.
- [ ] Run the local gates:

  ```bash
  uv run ruff format --check src tests
  uv run ruff check src tests
  uv run mypy
  uv run pytest -q -n auto
  ```

- [ ] Build and inspect both artifacts:

  ```bash
  uv build
  uv run --with twine twine check dist/*
  ```

  Confirm that the wheel contains `LICENSE`, `THIRD_PARTY_NOTICES.md`, vendored
  source and CUDA-extension source, but no compiled extensions, model weights,
  caches, or build directories.

## Publish

- [ ] Confirm CI is green on `main`.
- [ ] Create and push the new signed or annotated tag.
- [ ] Confirm the `wheels.yml` test, build, and trusted-publish jobs complete.
- [ ] Confirm the new version and files are visible on PyPI.
- [ ] Install the published wheel in a clean environment and run:

  ```bash
  python -c "import plumbline; print(plumbline.__version__)"
  plumbline doctor
  plumbline list-models
  plumbline list-datasets
  ```

- [ ] Create a GitHub Release from the CHANGELOG notes for the tag.

## Website and announcement

- [ ] Confirm the website deployment for the release commit succeeded.
- [ ] Decide whether to retain Basic Auth or make `plumbline-bench.org` public.
      Preserve the current audience unless the owner explicitly changes it.
- [ ] Add or update the PyPI and CI badges in the README if they are useful.
- [ ] Announce only after the package, documentation, and website show the same
      version and result count.

## Completed infrastructure

- [x] Repository is public.
- [x] `pypi` and `testpypi` GitHub environments exist.
- [x] PyPI trusted publishing works (`wheels.yml`).
- [x] The package metadata states that the wheel includes non-commercial code.
- [x] `LICENSE`, `THIRD_PARTY_NOTICES.md`, CHANGELOG, protocols, and reproduction
      files are bundled with the distribution.
- [x] CI runs lint, formatting, mypy, parallel tests, and site deployment.
- [x] Exact paper reruns have a separate `constraints/reproduction.txt` file;
      normal library installs accept the supported PyTorch range.
