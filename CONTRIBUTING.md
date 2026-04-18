# Contributing to plumbline

Thanks for the interest. `plumbline` aims to be the default reproducible eval
harness for 3D geometric foundation models, so the bar for correctness is
high — pull requests that change conventions, caching, reproduction configs,
or the JSON schema need an extra-careful review.

Everything else (new datasets, new models, metrics, docs) is straightforward
to contribute.

## Dev setup

```bash
uv sync                # creates .venv, installs dev deps
uv run pytest -q       # runs tests on CPU
uv run ruff check      # lint
uv run ruff format     # apply formatting
```

No GPU required for unit tests — the model-adapter suite is smoke tests only.

## Adding a model adapter

1. Pick a short kebab-case name (e.g. `vggt`, `mast3r`).
2. Create `src/plumbline/models/<name>.py` with:
   - A class subclassing `plumbline.models.base.Model`.
   - A `@register_model("<name>")` decorator.
   - Accurate `ModelCapabilities` (tasks, is_metric, view bounds, whether
     intrinsics are required).
   - Lazy torch import via `plumbline.models._torch_utils.ensure_torch()`.
   - `predict(images, intrinsics)` returning a `Prediction` in canonical
     conventions. Document every flip / transpose / scale / unit change with
     a comment citing the upstream source (paper section + repo file:line).
3. Add CLI-free smoke tests in `tests/test_model_adapters.py` covering:
   registration, instantiation without GPU, `config_hash` determinism, and
   view-bound enforcement.
4. Add the import path to `cli.py`'s eager-import list so the adapter shows
   up in `plumbline list-models`.

The adapter is responsible for resolution, normalization, device, and
convention conversion. The runner should never see torch.

## Adding a dataset loader

1. Create `src/plumbline/datasets/<name>.py` with:
   - A class subclassing `plumbline.datasets.base.Dataset`.
   - A `@register_dataset("<name>")` decorator.
   - A manifest-based scan so iteration doesn't touch the filesystem N^2.
   - A clear `DatasetNotAvailable` error pointing at the expected layout and
     URL when the root is missing.
2. Coordinate conversion happens **inside the loader**, exactly once, at load
   time. The runner must not touch coordinates.
3. First camera in every multi-view sample is the world frame
   (`rebase_to_first_camera`).
4. Add synthetic-fixture tests in `tests/test_datasets.py`.

## Adding a metric

Metrics are pure numpy, side-effect free. Inputs are canonical-convention
arrays; outputs are floats or dicts. Put them in `src/plumbline/metrics/`.

## Adding a reproduction

See [REPRODUCTIONS.md](./REPRODUCTIONS.md). A reproduction PR should include:

- The YAML config.
- A pinned sample list (if not already full-split).
- The exact paper citation and table/row.
- A first-run value with tolerance, committed to the YAML.

## Code style

- ruff enforces formatting + lint; CI will fail on violations.
- Type hints on public functions.
- Comments explain **why**, not what. Avoid narrating the code.

## Reporting bugs

Please include:
- Versions: `plumbline --version`, Python, torch, CUDA.
- The exact CLI command.
- A minimal repro (a small subset of the dataset + a small model variant is
  usually enough).
