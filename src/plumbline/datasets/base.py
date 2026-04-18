"""Base ``Dataset`` ABC and ``Sample`` dataclass.

Datasets iterate :class:`Sample` objects already in canonical conventions.
Coordinate conversion happens in the loader, exactly once, at load time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Dataset", "Sample"]


@dataclass
class Sample:
    """A single evaluation sample: one scene with ``N`` views of ground truth.

    Attributes
    ----------
    sample_id
        Stable, deterministic identifier. Used as a cache key and for logging.
    images
        ``(N, H, W, 3)`` uint8 sRGB.
    intrinsics
        ``(N, 3, 3)`` float32, pixel-space.
    extrinsics_gt
        ``(N, 4, 4)`` float32, ``world_from_camera``, first camera is world
        origin (see :func:`~plumbline.conventions.rebase_to_first_camera`).
    depth_gt
        Optional ``(N, H, W)`` float32 meters. NaN or 0 = invalid.
    depth_valid
        Optional ``(N, H, W)`` bool. If omitted, runner derives from
        :func:`~plumbline.conventions.depth_is_valid`.
    point_cloud_gt
        Optional ``(M, 3)`` world-frame ground-truth point cloud. Used by
        Chamfer/F-score metrics. May be subsampled for evaluation speed.
    metadata
        Free-form dict[str, Any]: scene id, split, difficulty, timestamps, etc.
    """

    sample_id: str
    images: NDArray[np.uint8]
    intrinsics: NDArray[np.float32]
    extrinsics_gt: NDArray[np.float32]
    depth_gt: NDArray[np.float32] | None = None
    depth_valid: NDArray[np.bool_] | None = None
    point_cloud_gt: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def num_views(self) -> int:
        return int(self.images.shape[0])


class Dataset(ABC):
    """Abstract base for plumbline datasets.

    Implementation rules (see :doc:`plan`):

    - Coordinate conversion happens once, in the loader. Never in the runner.
    - Pre-compute and cache a manifest (JSON) listing sample IDs and file
      paths. Iteration reads from the manifest, not from a directory scan.
    - Provide a ``subset(n)`` method for quick dev runs. Deterministic sampling
      (sort + stride, not random).
    - If the dataset requires auth/manual download, raise a clear error with
      URL and expected path layout on first use.
    """

    name: str
    split: str

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:  # pragma: no cover - abstract
        """Yield samples one at a time."""

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - abstract
        """Number of samples in the split."""

    def subset(self, n: int) -> Dataset:
        """Return a dataset of ``n`` deterministically chosen samples.

        Default implementation: stride-sample through ``self``. Subclasses
        may override for formats that support random access.
        """
        return _SubsetDataset(self, n)

    def subset_by_ids(self, sample_ids: list[str]) -> Dataset:
        """Return a dataset containing only the given ``sample_ids``, in order.

        Unknown IDs raise — a reproduction must fail loudly when the pinned
        sample list no longer matches the dataset on disk, since a silent
        drop would invalidate a paper-number comparison.
        """
        return _IdFilterDataset(self, sample_ids)


class _SubsetDataset(Dataset):
    """Stride-sampled subset wrapper.

    Indices are ``round(linspace(0, N-1, n))``. Stable across runs.
    """

    def __init__(self, source: Dataset, n: int) -> None:
        if n <= 0:
            raise ValueError(f"subset size must be > 0; got {n}")
        total = len(source)
        self._source = source
        n = min(n, total)
        self._indices = np.linspace(0, total - 1, n).round().astype(int).tolist()
        self.name = source.name
        self.split = f"{source.split}[subset={n}]"

    def __iter__(self) -> Iterator[Sample]:
        wanted = set(self._indices)
        for i, sample in enumerate(self._source):
            if i in wanted:
                yield sample
                wanted.discard(i)
                if not wanted:
                    return

    def __len__(self) -> int:
        return len(self._indices)


class _IdFilterDataset(Dataset):
    """Sample-ID-pinned subset wrapper.

    Iterates the source once, collecting samples whose ID is in the wanted
    set, and yields them in the order given by ``sample_ids``. Raises
    ``KeyError`` if any requested ID is absent.
    """

    def __init__(self, source: Dataset, sample_ids: list[str]) -> None:
        if not sample_ids:
            raise ValueError("sample_ids must be non-empty")
        self._source = source
        self._ids = list(sample_ids)
        self.name = source.name
        self.split = f"{source.split}[pinned={len(self._ids)}]"

    def __iter__(self) -> Iterator[Sample]:
        wanted = set(self._ids)
        collected: dict[str, Sample] = {}
        for sample in self._source:
            if sample.sample_id in wanted:
                collected[sample.sample_id] = sample
                if len(collected) == len(wanted):
                    break
        missing = [sid for sid in self._ids if sid not in collected]
        if missing:
            raise KeyError(
                f"{len(missing)} sample_id(s) from the pinned list were not "
                f"found in {self._source.name!r}: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        for sid in self._ids:
            yield collected[sid]

    def __len__(self) -> int:
        return len(self._ids)
