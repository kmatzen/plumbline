"""Prediction cache keyed by ``(model, dataset, sample_id, config_hash)``.

Inference is expensive; everything else is free. This cache lets the user:

- Re-run metrics with a different alignment mode without re-running inference.
- Change the report format without re-running inference.
- Invalidate per-model or per-dataset predictions without blowing away the
  whole cache.

On-disk layout::

    <cache_dir>/predictions/<model>/<config_hash>/<dataset>/<sample_id>.npz

Where:

- ``<model>`` is the model's ``name``.
- ``<config_hash>`` is ``model.config_hash()`` — typically includes version +
  preprocessing knobs (resolution, view count, precision).
- ``<dataset>`` is the dataset's ``name``.
- ``<sample_id>`` is the ``Sample.sample_id``, sanitized for filesystems.

Each ``.npz`` stores the non-None fields of a :class:`Prediction`, plus a
``metadata`` JSON blob under key ``_meta_json``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from plumbline.models.base import Prediction

__all__ = [
    "DEFAULT_CACHE_DIR",
    "PredictionCache",
    "default_cache_dir",
    "sanitize_id",
]


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_id(sample_id: str) -> str:
    """Replace characters that are awkward on Windows/POSIX filesystems.

    Preserves dots, dashes, underscores, and ASCII alphanumerics. Any run of
    other characters collapses to a single underscore. Long IDs are truncated
    and suffixed with a short hash to stay under 128 bytes.
    """
    cleaned = _UNSAFE.sub("_", sample_id.strip())
    if not cleaned:
        cleaned = "_"
    if len(cleaned) > 128:
        tail = hashlib.sha256(sample_id.encode()).hexdigest()[:8]
        cleaned = cleaned[:119] + "_" + tail
    return cleaned


def default_cache_dir() -> Path:
    """Where the cache lives by default.

    Honors ``PLUMBLINE_CACHE_DIR`` if set, else ``~/.cache/plumbline``.
    """
    override = os.environ.get("PLUMBLINE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "plumbline"


DEFAULT_CACHE_DIR: Path = default_cache_dir()


class PredictionCache:
    """On-disk cache of :class:`Prediction` objects."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.predictions_dir = self.root / "predictions"

    def path_for(
        self,
        model_name: str,
        config_hash: str,
        dataset_name: str,
        sample_id: str,
        input_fingerprint: str = "",
    ) -> Path:
        """Return the on-disk path for a prediction entry.

        ``input_fingerprint`` is a short hash of the loader's actual
        preprocessing output (see ``runner._sample_input_fingerprint``).
        Mixing it into the path means a loader refactor — e.g. a new
        image resolution or warp — stores to a distinct file so stale
        predictions can't be served against fresh GT (D21, 2026-04-24).
        Empty string preserves the pre-fingerprint cache layout for
        callers that predate the fingerprint (e.g. ``clear``, tests).
        """
        suffix = f"_{input_fingerprint}" if input_fingerprint else ""
        return (
            self.predictions_dir
            / _slug(model_name)
            / _slug(config_hash)
            / _slug(dataset_name)
            / f"{sanitize_id(sample_id)}{suffix}.npz"
        )

    def has(
        self,
        model_name: str,
        config_hash: str,
        dataset_name: str,
        sample_id: str,
        input_fingerprint: str = "",
    ) -> bool:
        return self.path_for(
            model_name, config_hash, dataset_name, sample_id, input_fingerprint
        ).exists()

    def save(
        self,
        model_name: str,
        config_hash: str,
        dataset_name: str,
        sample_id: str,
        prediction: Prediction,
        *,
        input_fingerprint: str = "",
    ) -> Path:
        """Persist a prediction to disk as compressed NPZ.

        Only non-None array fields are stored; ``metadata`` becomes a JSON blob.
        """
        path = self.path_for(model_name, config_hash, dataset_name, sample_id, input_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        for key in ("depth", "intrinsics", "extrinsics", "point_map", "confidence"):
            value = getattr(prediction, key)
            if value is not None:
                arrays[key] = np.asarray(value)

        meta = prediction.metadata or {}
        arrays["_meta_json"] = np.asarray(json.dumps(meta, default=_json_default))

        # np.savez_compressed appends ".npz"; write to a distinct basename
        # and move into place atomically. The numpy stub claims the second
        # positional arg is `bool`, but runtime accepts **arrays; suppress.
        tmp = path.with_name(path.stem + ".tmp")
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        os.replace(tmp.with_suffix(".tmp.npz"), path)
        return path

    def load(
        self,
        model_name: str,
        config_hash: str,
        dataset_name: str,
        sample_id: str,
        input_fingerprint: str = "",
    ) -> Prediction:
        path = self.path_for(model_name, config_hash, dataset_name, sample_id, input_fingerprint)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            kwargs: dict[str, object] = {}
            for key in ("depth", "intrinsics", "extrinsics", "point_map", "confidence"):
                if key in data.files:
                    kwargs[key] = data[key]
            meta_raw = data["_meta_json"] if "_meta_json" in data.files else None
        meta: dict[str, object] = {}
        if meta_raw is not None:
            meta = json.loads(str(meta_raw))
        return Prediction(metadata=meta, **kwargs)  # type: ignore[arg-type]

    def clear(
        self,
        *,
        model: str | None = None,
        config_hash: str | None = None,
        dataset: str | None = None,
    ) -> int:
        """Remove cached predictions matching the given filters.

        Each filter narrows the set of removed entries. ``dataset=X`` with no
        ``model`` removes that dataset's predictions across all models.
        Returns the number of ``.npz`` files deleted.
        """
        if not self.predictions_dir.exists():
            return 0

        # Enumerate the (model, config) roots we might touch.
        if model:
            model_roots: Iterable[Path] = [self.predictions_dir / _slug(model)]
        else:
            model_roots = [p for p in self.predictions_dir.iterdir() if p.is_dir()]

        removed = 0
        for m_root in model_roots:
            if not m_root.exists():
                continue

            if config_hash:
                config_roots: Iterable[Path] = [m_root / _slug(config_hash)]
            else:
                config_roots = [p for p in m_root.iterdir() if p.is_dir()]

            for c_root in config_roots:
                if not c_root.exists():
                    continue
                if dataset:
                    removed += _rmtree_count(c_root / _slug(dataset))
                else:
                    removed += _rmtree_count(c_root)

            # Clean up empty model dir.
            if m_root.exists() and not any(m_root.iterdir()):
                with contextlib.suppress(OSError):
                    m_root.rmdir()

        return removed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    """Slug safe for use as a directory name."""
    s = _UNSAFE.sub("_", s.strip()) or "_"
    return s[:128]


def _rmtree_count(path: Path) -> int:
    """Remove ``path`` recursively; return count of ``.npz`` files deleted."""
    if not path.exists():
        return 0
    count = 0
    if path.is_file():
        if path.suffix == ".npz":
            count = 1
        path.unlink()
        return count
    for sub in sorted(path.rglob("*.npz")):
        try:
            sub.unlink()
            count += 1
        except FileNotFoundError:
            pass
    # Clean up empty directories.
    for sub in sorted(path.rglob("*"), reverse=True):
        if sub.is_dir():
            with contextlib.suppress(OSError):
                sub.rmdir()
    with contextlib.suppress(OSError):
        path.rmdir()
    return count


def _json_default(obj: object) -> object:
    """Coerce unknown types to JSON-safe values (numpy scalars, paths)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


# Avoid unused-import warning if users rely on ``asdict`` only indirectly.
_ = asdict
