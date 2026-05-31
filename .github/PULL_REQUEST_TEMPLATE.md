<!-- Thanks for contributing to plumbline! See CONTRIBUTING.md for the full bar. -->

## What & why

<!-- What does this change and why? Link any related issue (e.g. Closes #123). -->

## Type of change

- [ ] New model adapter
- [ ] New dataset loader
- [ ] New metric
- [ ] New / updated reproduction
- [ ] Bug fix
- [ ] Docs / infra
- [ ] Other:

## Checklist

- [ ] `uv run ruff check src tests` passes
- [ ] `uv run ruff format --check src tests` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest -q` passes
- [ ] Coordinate / unit / scale conversions are documented with a comment
      citing the upstream source (paper section + repo `file:line`)

## For reproduction PRs

<!-- Delete this section if not applicable. -->

- [ ] YAML declares `protocol: <name>` (or adds a preset with a paper-cited header)
- [ ] `paper_reference` has the exact **table + column + row** and
      `source_confidence: verified_pdf`, verified against the arXiv PDF (not a summary)
- [ ] Pinned sample list committed in-repo (if the loader has no deterministic default)
- [ ] First-run observed value + tolerance committed to the YAML
- [ ] `REPRODUCTIONS.md` and the site count updated if a new ✅ cell lands
