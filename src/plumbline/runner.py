"""Evaluation runner.

Pulls samples from a :class:`Dataset`, runs inference via a :class:`Model`
(caching raw predictions to disk), computes metrics, and returns a
:class:`Report`.

Design rules
------------
- Raw predictions are cached **before** any alignment or resize. Changing
  alignment or metrics must not trigger re-inference.
- Predictions at model-native resolution are resized to GT resolution for
  metric computation; GT is never downsampled to the prediction resolution.
- OOM on a single sample causes a skip, not a crash. The run continues.
- Determinism: seed numpy and torch (when available); log GPU, CUDA, and
  checkpoint hash.
"""

from __future__ import annotations

import logging
import math
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline import _version
from plumbline.cache import PredictionCache
from plumbline.conventions import depth_is_valid
from plumbline.datasets.base import Dataset, Sample
from plumbline.metrics.alignment import align_depth
from plumbline.metrics.depth import abs_rel, delta_threshold, log10_error, rmse, silog
from plumbline.metrics.pointmap import chamfer_distance, f_score
from plumbline.metrics.pose import auc as pose_auc_fn
from plumbline.metrics.pose import rotation_error_degrees, translation_cosine_error
from plumbline.models.base import Model, Prediction
from plumbline.report import Report, RunEnvironment, SampleResult

__all__ = ["EvaluateConfig", "evaluate"]

log = logging.getLogger("plumbline.runner")


@dataclass
class EvaluateConfig:
    """Runner configuration.

    Every field here factors into the cache-key config hash when stored on
    disk, so changing one re-triggers inference. Pure metric/report knobs
    (alignment, format) do not live in this dataclass.
    """

    tasks: list[str]
    max_views: int = 8
    device: str = "cuda:0"
    seed: int = 0


def evaluate(
    model: Model,
    dataset: Dataset,
    tasks: list[str],
    *,
    scale_alignment: str = "median",
    max_views: int = 8,
    device: str = "cuda:0",
    cache: PredictionCache | None = None,
    seed: int = 0,
    pose_auc_thresholds: tuple[float, ...] = (5.0, 10.0, 30.0),
    delta_thresholds: tuple[float, ...] = (1.25, 1.25**2, 1.25**3),
    f_score_threshold: float = 0.05,
) -> Report:
    """Evaluate a model on a dataset and return a :class:`Report`.

    Parameters mirror the CLI's `run` command. See module docstring for the
    contract on caching, resolution, and OOM handling.
    """
    _seed_everything(seed)
    cache = cache or PredictionCache()

    unknown = [t for t in tasks if not model.capabilities.supports_task(t)]
    if unknown:
        raise ValueError(
            f"Model '{model.name}' does not support tasks {unknown}; "
            f"capabilities: {sorted(model.capabilities.tasks)}"
        )

    report = Report(
        model=model.name,
        model_version=getattr(model, "version", ""),
        dataset=dataset.name,
        split=getattr(dataset, "split", ""),
        tasks=list(tasks),
        scale_alignment=scale_alignment,
        aggregate_metrics={},
        config_hash=model.config_hash(),
        environment=_detect_environment(),
    )

    per_metric_values: dict[str, list[float]] = {}

    total = _safe_len(dataset)
    report.n_total = total if total is not None else 0

    for sample in dataset:
        report.n_total = max(report.n_total, len(report.per_sample) + 1)
        t0 = time.perf_counter()
        prediction = _predict_with_cache(
            model=model,
            dataset_name=dataset.name,
            sample=sample,
            max_views=max_views,
            cache=cache,
        )
        if prediction is None:
            report.n_skipped += 1
            report.per_sample.append(
                SampleResult(
                    sample_id=sample.sample_id,
                    metrics={},
                    skipped=True,
                    skip_reason="OOM or adapter error (see logs)",
                )
            )
            continue

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        sample_metrics = _compute_metrics(
            prediction=prediction,
            sample=sample,
            tasks=tasks,
            scale_alignment=scale_alignment,
            pose_auc_thresholds=pose_auc_thresholds,
            delta_thresholds=delta_thresholds,
            f_score_threshold=f_score_threshold,
        )
        report.per_sample.append(
            SampleResult(
                sample_id=sample.sample_id,
                metrics=sample_metrics,
                runtime_ms=runtime_ms,
            )
        )
        report.n_evaluated += 1
        for key, value in sample_metrics.items():
            per_metric_values.setdefault(key, []).append(value)

    # Aggregate with per-sample mean; skip NaNs.
    aggregate: dict[str, float] = {}
    for key, values in per_metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        aggregate[key] = float(arr.mean()) if arr.size else float("nan")
    report.aggregate_metrics = aggregate
    return report


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _predict_with_cache(
    *,
    model: Model,
    dataset_name: str,
    sample: Sample,
    max_views: int,
    cache: PredictionCache,
) -> Prediction | None:
    cache_args = (model.name, model.config_hash(), dataset_name, sample.sample_id)
    if cache.has(*cache_args):
        try:
            return cache.load(*cache_args)
        except Exception:
            log.warning("Cache load failed for %s; re-running", sample.sample_id, exc_info=True)

    # Trim to the configured max_views if the model's cap permits a smaller set.
    n = sample.images.shape[0]
    model_cap = model.capabilities.max_views
    cap = n if math.isinf(model_cap) else int(model_cap)
    keep = min(n, max_views, cap)
    if keep < int(model.capabilities.min_views):
        log.warning(
            "Skipping %s: dataset provides %d views, model '%s' needs >= %d",
            sample.sample_id,
            n,
            model.name,
            model.capabilities.min_views,
        )
        return None

    images = sample.images[:keep]
    intrinsics = (
        sample.intrinsics[:keep].astype(np.float32)
        if model.capabilities.requires_intrinsics
        else None
    )

    try:
        prediction = model.predict(images, intrinsics=intrinsics)
    except _OOM_TYPES as exc:  # pragma: no cover — needs CUDA to exercise
        log.warning("OOM on %s: %s", sample.sample_id, exc)
        _try_empty_cuda_cache()
        return None
    except Exception:
        log.error(
            "Adapter '%s' failed on %s:\n%s",
            model.name,
            sample.sample_id,
            traceback.format_exc(),
        )
        return None

    try:
        cache.save(*cache_args, prediction=prediction)
    except OSError:
        log.warning("Failed to cache prediction for %s", sample.sample_id, exc_info=True)

    return prediction


def _compute_metrics(
    *,
    prediction: Prediction,
    sample: Sample,
    tasks: list[str],
    scale_alignment: str,
    pose_auc_thresholds: tuple[float, ...],
    delta_thresholds: tuple[float, ...],
    f_score_threshold: float,
) -> dict[str, float]:
    out: dict[str, float] = {}

    wants_depth = "mono_depth" in tasks or "mvs_depth" in tasks
    if wants_depth and prediction.depth is not None and sample.depth_gt is not None:
        out.update(
            _depth_metrics(
                pred=prediction.depth,
                gt=sample.depth_gt,
                valid=sample.depth_valid,
                scale_alignment=scale_alignment,
                delta_thresholds=delta_thresholds,
            )
        )

    if "pose" in tasks and prediction.extrinsics is not None:
        out.update(_pose_metrics(prediction.extrinsics, sample.extrinsics_gt, pose_auc_thresholds))

    # Point-cloud metrics fire whenever both are present; gated on
    # "mvs_depth" or "point_cloud" to keep mono-depth runs cheap.
    wants_pcd = "mvs_depth" in tasks or "point_cloud" in tasks
    if wants_pcd and prediction.point_map is not None and sample.point_cloud_gt is not None:
        out.update(
            _point_cloud_metrics(
                point_map=prediction.point_map,
                point_cloud_gt=sample.point_cloud_gt,
                f_score_threshold=f_score_threshold,
            )
        )

    return out


def _depth_metrics(
    *,
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
    scale_alignment: str,
    delta_thresholds: tuple[float, ...],
) -> dict[str, float]:
    pred_aligned, gt_flat, valid_flat = _flatten_pred_gt(pred, gt, valid)
    if pred_aligned.size == 0:
        return {k: float("nan") for k in _depth_metric_keys(delta_thresholds)}

    aligned = align_depth(pred_aligned, gt_flat, valid_flat, mode=scale_alignment)

    metrics: dict[str, float] = {
        "abs_rel": abs_rel(aligned, gt_flat, valid_flat),
        "rmse": rmse(aligned, gt_flat, valid_flat),
        "log10": log10_error(aligned, gt_flat, valid_flat),
        "silog": silog(aligned, gt_flat, valid_flat),
    }
    for i, t in enumerate(delta_thresholds, start=1):
        metrics[f"delta_{i}"] = delta_threshold(aligned, gt_flat, valid_flat, threshold=t)
    return metrics


def _depth_metric_keys(deltas: tuple[float, ...]) -> list[str]:
    return ["abs_rel", "rmse", "log10", "silog", *[f"delta_{i}" for i in range(1, len(deltas) + 1)]]


def _flatten_pred_gt(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Resize prediction to GT resolution per-view and flatten batches."""
    pred_out = pred.astype(np.float64) if pred.shape == gt.shape else _resize_depth_to_gt(pred, gt)
    if valid is None:
        valid = depth_is_valid(gt)
    return pred_out, gt.astype(np.float64), valid


def _resize_depth_to_gt(pred: NDArray[Any], gt: NDArray[Any]) -> NDArray[Any]:
    """Resize a batch of depth maps to GT resolution with bilinear sampling.

    Never bilinear-resizes ground-truth; the runner only resizes predictions.
    Uses PIL so it runs on numpy alone, no torch dependency.
    """
    from PIL import Image

    if pred.shape[-2:] == gt.shape[-2:]:
        return pred.astype(np.float64)
    if pred.ndim != gt.ndim:
        raise ValueError(f"ndim mismatch: pred {pred.shape} vs gt {gt.shape}")
    batch_shape = pred.shape[:-2]
    tgt_h, tgt_w = gt.shape[-2:]
    flat = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    out = np.empty((flat.shape[0], tgt_h, tgt_w), dtype=np.float64)
    for i, m in enumerate(flat):
        img = Image.fromarray(m.astype(np.float32), mode="F")
        resample = Image.Resampling.BILINEAR
        out[i] = np.asarray(img.resize((tgt_w, tgt_h), resample=resample), dtype=np.float64)
    return out.reshape(*batch_shape, tgt_h, tgt_w)


def _point_cloud_metrics(
    *,
    point_map: NDArray[Any],
    point_cloud_gt: NDArray[Any],
    f_score_threshold: float,
) -> dict[str, float]:
    """Chamfer + F-score between the flattened prediction and GT point cloud.

    Flattens ``(N, H, W, 3)`` prediction into a single ``(M, 3)`` cloud,
    dropping NaNs/zeros, and compares against the GT cloud with the given
    distance threshold (meters, matching the units of the point map).
    """
    if point_map.ndim < 2 or point_map.shape[-1] != 3:
        raise ValueError(f"point_map must end in 3 for xyz; got {point_map.shape}")
    pts = point_map.reshape(-1, 3).astype(np.float64)
    # Drop NaN-marked invalid points.
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        return {
            "chamfer": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f_score": float("nan"),
        }

    chamfer = chamfer_distance(pts.astype(np.float32), point_cloud_gt.astype(np.float32))
    f = f_score(
        pts.astype(np.float32),
        point_cloud_gt.astype(np.float32),
        threshold=f_score_threshold,
    )
    return {"chamfer": float(chamfer), **{k: float(v) for k, v in f.items()}}


def _pose_metrics(
    E_pred: NDArray[Any],
    E_gt: NDArray[Any],
    auc_thresholds: tuple[float, ...],
) -> dict[str, float]:
    if E_pred.shape != E_gt.shape:
        raise ValueError(f"pose shape mismatch: {E_pred.shape} vs {E_gt.shape}")
    # Skip camera 0 (identity by convention, no error to compute).
    if E_pred.ndim == 3 and E_pred.shape[0] > 1:
        Ep = E_pred[1:]
        Eg = E_gt[1:]
    else:
        Ep = E_pred
        Eg = E_gt
    rot = rotation_error_degrees(Ep, Eg)
    trans = translation_cosine_error(Ep[..., :3, 3], Eg[..., :3, 3])
    combined = np.maximum(np.asarray(rot).reshape(-1), np.asarray(trans).reshape(-1))
    aucs = pose_auc_fn(combined, list(auc_thresholds))
    out: dict[str, float] = {
        "rotation_error_deg_mean": float(np.nanmean(np.asarray(rot))),
        "translation_cos_err_deg_mean": float(np.nanmean(np.asarray(trans))),
    }
    for t, v in aucs.items():
        out[f"pose_auc@{t:g}"] = float(v)
    return out


def _detect_environment() -> RunEnvironment:
    env = RunEnvironment(plumbline_version=_version.__version__)
    try:
        import torch

        env.torch_version = torch.__version__
        if torch.cuda.is_available():
            env.cuda_version = torch.version.cuda
            env.gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _safe_len(dataset: Dataset) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def _try_empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Types we treat as OOM. Lazy-resolved so `import plumbline.runner` doesn't
# require torch installed.
def _oom_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [MemoryError]
    try:
        import torch

        types.append(torch.cuda.OutOfMemoryError)
    except Exception:
        pass
    return tuple(types)


_OOM_TYPES = _oom_types()
