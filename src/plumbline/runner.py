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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from plumbline import _version
from plumbline.cache import PredictionCache
from plumbline.conventions import depth_is_valid
from plumbline.datasets.base import Dataset, Sample
from plumbline.metrics.alignment import (
    align_depth,
    apply_similarity,
    icp_similarity,
    umeyama_similarity,
)
from plumbline.metrics.depth import abs_rel, delta_threshold, log10_error, rmse, silog
from plumbline.metrics.pointmap import (
    accuracy_completeness,
    chamfer_distance,
    f_score,
)
from plumbline.metrics.pose import auc as pose_auc_fn
from plumbline.metrics.pose import (
    pairwise_pose_errors,
    rotation_error_degrees,
    translation_cosine_error,
)
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
    depth_clip: tuple[float, float] | None = None,
    pointcloud_alignment: str = "none",
    chamfer_outlier_distance: float | None = None,
    mask_boundaries: bool = False,
    boundary_thickness: int = 1,
    boundary_tol: float = 0.1,
    aggregation: str = "sample",
    scene_voxel_size: float = 0.01,
    show_progress: bool = True,
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

    # Scene-merge mode accumulates aligned point clouds by scene prefix
    # and computes Acc/Comp/Overall on the merged cloud at the end —
    # matches the ETH3D / MVS paper protocol where predictions across all
    # views are fused into one reconstruction before comparison with GT.
    if aggregation not in ("sample", "scene"):
        raise ValueError(f"unknown aggregation '{aggregation}'; use 'sample' or 'scene'")
    scene_points: dict[str, list[NDArray[np.float32]]] = {}
    scene_gt: dict[str, NDArray[Any]] = {}

    # Progress bar on the main sample loop. Writes to stderr so stdout
    # (the final Report markdown) stays clean. Set show_progress=False
    # for programmatic callers that render their own UI.
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("  {task.fields[status]}"),
        console=Console(stderr=True),
        disable=not show_progress,
        transient=False,
    )
    task_total = total if total is not None else None
    task_desc = f"{model.name}/{dataset.name}"
    progress_task = progress.add_task(task_desc, total=task_total, status="")

    def _iter_with_progress(ds: Dataset) -> Iterator[Sample]:
        """Advance the progress bar AFTER each iteration regardless of
        whether the loop body `continue`d or fell through."""
        for s in ds:
            yield s
            # Running status: one primary metric's mean, if any were
            # computed. Choose first metric alphabetically for stability.
            status = ""
            if per_metric_values:
                key = sorted(per_metric_values.keys())[0]
                vals = [v for v in per_metric_values[key] if np.isfinite(v)]
                if vals:
                    status = f"{key}={float(np.mean(vals)):.4f}  skipped={report.n_skipped}"
            progress.update(progress_task, advance=1, status=status)

    progress.start()

    for sample in _iter_with_progress(dataset):
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
        # The runner trimmed the model input to ``keep`` views in
        # ``_predict_with_cache`` (honouring max_views + the model's cap).
        # Trim the sample's per-view GT arrays to the same count so that
        # pose / depth / point-map metrics see matching shapes. Sample IDs
        # that genuinely need all-view GT should raise max_views accordingly.
        trimmed_sample = _trim_sample_to_views(
            sample,
            prediction.extrinsics.shape[0]
            if prediction.extrinsics is not None
            else (
                prediction.depth.shape[0]
                if prediction.depth is not None
                else sample.images.shape[0]
            ),
        )
        if aggregation == "scene" and trimmed_sample.point_cloud_gt is not None:
            aligned = _aligned_point_map(
                prediction=prediction,
                sample=trimmed_sample,
                pointcloud_alignment=pointcloud_alignment,
            )
            if aligned is not None:
                scene = sample.sample_id.split("/", 1)[0]
                scene_points.setdefault(scene, []).append(aligned.reshape(-1, 3))
                # GT is the same across samples of a scene (subsampled
                # laser scan); keep the first one we see.
                scene_gt.setdefault(scene, trimmed_sample.point_cloud_gt)
                report.per_sample.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        metrics={"n_points": float(aligned.size // 3)},
                        runtime_ms=runtime_ms,
                    )
                )
                report.n_evaluated += 1
            else:
                report.n_skipped += 1
                report.per_sample.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        metrics={},
                        skipped=True,
                        skip_reason="no point map (missing depth or K/E)",
                    )
                )
            continue

        sample_metrics = _compute_metrics(
            prediction=prediction,
            sample=trimmed_sample,
            tasks=tasks,
            scale_alignment=scale_alignment,
            pose_auc_thresholds=pose_auc_thresholds,
            delta_thresholds=delta_thresholds,
            f_score_threshold=f_score_threshold,
            depth_clip=depth_clip,
            pointcloud_alignment=pointcloud_alignment,
            chamfer_outlier_distance=chamfer_outlier_distance,
            mask_boundaries=mask_boundaries,
            boundary_thickness=boundary_thickness,
            boundary_tol=boundary_tol,
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

    progress.stop()

    # Scene-merge post-processing: merge per-scene, voxel downsample,
    # compute Acc/Comp/Overall against scene GT. Aggregate across scenes
    # with an unweighted mean — matches the per-scene-then-average
    # convention in MVS benchmark tables.
    if aggregation == "scene" and scene_points:
        scene_progress = Progress(
            TextColumn("[bold blue]scene-agg"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("  {task.fields[status]}"),
            console=Console(stderr=True),
            disable=not show_progress,
            transient=False,
        )
        scene_task = scene_progress.add_task(
            "chamfer", total=len(scene_points), status=""
        )
        scene_progress.start()
        per_scene: dict[str, dict[str, float]] = {}
        for scene, chunks in scene_points.items():
            scene_progress.update(scene_task, status=f"scene={scene}")
            merged = np.vstack(chunks).astype(np.float32)
            gt = scene_gt[scene]
            per_scene[scene] = accuracy_completeness(
                merged, gt, voxel_size=scene_voxel_size
            )
            scene_progress.advance(scene_task, 1)
        scene_progress.stop()
        report.per_scene_metrics = per_scene
        keys = sorted({k for m in per_scene.values() for k in m})
        report.aggregate_metrics = {
            k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys
        }
        return report

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
    depth_clip: tuple[float, float] | None = None,
    pointcloud_alignment: str = "none",
    chamfer_outlier_distance: float | None = None,
    mask_boundaries: bool = False,
    boundary_thickness: int = 1,
    boundary_tol: float = 0.1,
) -> dict[str, float]:
    out: dict[str, float] = {}

    wants_depth = "mono_depth" in tasks or "mvs_depth" in tasks
    if wants_depth and prediction.depth is not None and sample.depth_gt is not None:
        valid = sample.depth_valid
        if mask_boundaries:
            # MoGe / Depth Pro / many mono-depth papers exclude pixels near
            # GT depth discontinuities from evaluation (sensor noise at
            # edges). Port of depth_occlusion_edge_numpy: disparity-space
            # max/min-over-window edge detection + dilation + AND-fg-bg.
            valid = _apply_boundary_mask(
                gt=sample.depth_gt,
                valid=valid,
                thickness=boundary_thickness,
                tol=boundary_tol,
            )
        out.update(
            _depth_metrics(
                pred=prediction.depth,
                gt=sample.depth_gt,
                valid=valid,
                scale_alignment=scale_alignment,
                delta_thresholds=delta_thresholds,
                depth_clip=depth_clip,
            )
        )

    if "pose" in tasks and prediction.extrinsics is not None:
        out.update(_pose_metrics(prediction.extrinsics, sample.extrinsics_gt, pose_auc_thresholds))

    # Point-cloud metrics fire whenever both are present; gated on
    # "mvs_depth" or "point_cloud" to keep mono-depth runs cheap.
    wants_pcd = "mvs_depth" in tasks or "point_cloud" in tasks
    if wants_pcd and sample.point_cloud_gt is not None:
        pmap = _aligned_point_map(
            prediction=prediction, sample=sample, pointcloud_alignment=pointcloud_alignment
        )
        if pmap is None:
            return out
        out.update(
            _point_cloud_metrics(
                point_map=pmap,
                point_cloud_gt=sample.point_cloud_gt,
                f_score_threshold=f_score_threshold,
                outlier_distance=chamfer_outlier_distance,
            )
        )

    return out


def _aligned_point_map(
    *,
    prediction: Prediction,
    sample: Sample,
    pointcloud_alignment: str,
) -> NDArray[np.float32] | None:
    """Build the aligned dense point map for ``prediction`` in the GT scene frame.

    Used by both per-sample chamfer/F-score metrics and the scene-merge
    aggregation path. Returns ``None`` when there isn't enough information
    (no point_map and no depth+K+E).
    """
    pmap = prediction.point_map
    if pmap is None:
        if (
            prediction.depth is None
            or prediction.intrinsics is None
            or prediction.extrinsics is None
        ):
            return None
        pmap = _back_project_depth(prediction.depth, prediction.intrinsics, prediction.extrinsics)
    if pointcloud_alignment == "camera_centers":
        if prediction.extrinsics is not None and sample.extrinsics_gt is not None:
            pred_centers = prediction.extrinsics[:, :3, 3]
            gt_centers = sample.extrinsics_gt[:, :3, 3]
            if pred_centers.shape[0] >= 3:
                s, R, t = umeyama_similarity(pred_centers, gt_centers)
                pmap = apply_similarity(pmap, s, R, t).astype(np.float32)
            else:
                log.warning(
                    "pointcloud_alignment=camera_centers needs >= 3 views; "
                    "got %d — leaving point map unaligned",
                    int(pred_centers.shape[0]),
                )
    elif pointcloud_alignment == "icp":
        warm_s = warm_R = warm_t = None
        if prediction.extrinsics is not None and sample.extrinsics_gt is not None:
            pred_centers = prediction.extrinsics[:, :3, 3]
            gt_centers = sample.extrinsics_gt[:, :3, 3]
            if pred_centers.shape[0] >= 3:
                warm_s, warm_R, warm_t = umeyama_similarity(pred_centers, gt_centers)
        s, R, t, _info = icp_similarity(
            pmap.reshape(-1, 3),
            sample.point_cloud_gt,
            init_s=warm_s,
            init_R=warm_R,
            init_t=warm_t,
        )
        pmap = apply_similarity(pmap, s, R, t).astype(np.float32)
    elif pointcloud_alignment != "none":
        raise ValueError(
            f"unknown pointcloud_alignment '{pointcloud_alignment}'; "
            "use 'none', 'camera_centers', or 'icp'"
        )
    return pmap


def _trim_sample_to_views(sample: Sample, n: int) -> Sample:
    """Return a new Sample whose per-view arrays are sliced to the first ``n`` views.

    Adapters that cap ``max_views`` (MASt3R's 2-view PairViewer, VGGT's
    N<=32) leave a predicted tensor with fewer views than the loader
    produced. Pose / depth / point-map metrics expect matching view
    counts; this helper makes the mismatch a non-issue. No-op when the
    sample already has ≤ n views.
    """
    if n <= 0 or sample.images.shape[0] <= n:
        return sample
    from dataclasses import replace

    return replace(
        sample,
        images=sample.images[:n],
        intrinsics=sample.intrinsics[:n],
        extrinsics_gt=sample.extrinsics_gt[:n],
        depth_gt=sample.depth_gt[:n] if sample.depth_gt is not None else None,
        depth_valid=sample.depth_valid[:n] if sample.depth_valid is not None else None,
        # point_cloud_gt is scene-level (M, 3); not per-view, so leave it.
    )


def _depth_metrics(
    *,
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
    scale_alignment: str,
    delta_thresholds: tuple[float, ...],
    depth_clip: tuple[float, float] | None = None,
) -> dict[str, float]:
    pred_aligned, gt_flat, valid_flat = _flatten_pred_gt(pred, gt, valid)
    if pred_aligned.size == 0:
        return {k: float("nan") for k in _depth_metric_keys(delta_thresholds)}

    aligned = align_depth(pred_aligned, gt_flat, valid_flat, mode=scale_alignment)
    # Post-alignment clip to a physical depth range. Standard NYU/KITTI eval
    # protocol clips predictions to the same range as the valid-GT mask so a
    # handful of alignment-induced outliers don't pull the mean AbsRel up by
    # orders of magnitude (seen on DA-V2 Large sample 88 where aligned
    # depth blew up to 1e8m without this clip).
    if depth_clip is not None:
        lo, hi = depth_clip
        aligned = np.clip(aligned, lo, hi)

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
    outlier_distance: float | None = None,
) -> dict[str, float]:
    """Chamfer + F-score between the flattened prediction and GT point cloud.

    Flattens ``(N, H, W, 3)`` prediction into a single ``(M, 3)`` cloud,
    dropping NaNs/zeros, and compares against the GT cloud with the given
    distance threshold (meters, matching the units of the point map).

    Parameters
    ----------
    outlier_distance
        If set, discard predicted points farther than this many units from
        their nearest GT neighbour before computing chamfer + precision.
        Matches the ETH3D / Tanks & Temples paper protocol, where the
        reported chamfer/F-score is on the inlier pool so a tiny fraction
        of hallucinations can't dominate the metric.
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

    chamfer = chamfer_distance(
        pts.astype(np.float32),
        point_cloud_gt.astype(np.float32),
        outlier_distance=outlier_distance,
    )
    f = f_score(
        pts.astype(np.float32),
        point_cloud_gt.astype(np.float32),
        threshold=f_score_threshold,
        outlier_distance=outlier_distance,
    )
    return {"chamfer": float(chamfer), **{k: float(v) for k, v in f.items()}}


def _pose_metrics(
    E_pred: NDArray[Any],
    E_gt: NDArray[Any],
    auc_thresholds: tuple[float, ...],
) -> dict[str, float]:
    if E_pred.shape != E_gt.shape:
        raise ValueError(f"pose shape mismatch: {E_pred.shape} vs {E_gt.shape}")
    # Absolute per-view errors: compare pred[i] vs GT[i] in their shared
    # world frame (view-0 identity by convention). Skip camera 0.
    if E_pred.ndim == 3 and E_pred.shape[0] > 1:
        Ep = E_pred[1:]
        Eg = E_gt[1:]
    else:
        Ep = E_pred
        Eg = E_gt
    abs_rot = np.asarray(rotation_error_degrees(Ep, Eg)).reshape(-1)
    abs_trans = np.asarray(translation_cosine_error(Ep[..., :3, 3], Eg[..., :3, 3])).reshape(-1)
    abs_combined = np.maximum(abs_rot, abs_trans)
    abs_aucs = pose_auc_fn(abs_combined, list(auc_thresholds))

    out: dict[str, float] = {
        "rotation_error_deg_mean": float(np.nanmean(abs_rot)),
        "translation_cos_err_deg_mean": float(np.nanmean(abs_trans)),
    }
    for t, v in abs_aucs.items():
        out[f"pose_auc@{t:g}"] = float(v)

    # Pairwise relative pose errors: frame-invariant, compared over all
    # N*(N-1)/2 unordered pairs. This is the metric VGGT / MASt3R /
    # DUSt3R papers report — directly comparable to their tables.
    if E_pred.ndim == 3 and E_pred.shape[0] >= 2:
        pw_rot, pw_trans = pairwise_pose_errors(E_pred, E_gt)
        pw_combined = np.maximum(pw_rot, pw_trans)
        pw_aucs = pose_auc_fn(pw_combined, list(auc_thresholds))
        out["pairwise_rot_err_deg_mean"] = float(np.nanmean(pw_rot))
        out["pairwise_trans_cos_err_deg_mean"] = float(np.nanmean(pw_trans))
        for t, v in pw_aucs.items():
            out[f"pairwise_pose_auc@{t:g}"] = float(v)

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


def _apply_boundary_mask(
    *,
    gt: NDArray[Any],
    valid: NDArray[Any] | None,
    thickness: int,
    tol: float,
) -> NDArray[np.bool_]:
    """AND MoGe's boundary-edge mask into the existing valid mask.

    Operates on ``(N, H, W)`` or ``(H, W)`` GT depth. Returns a bool
    array of the same shape with ``False`` where the pixel either
    wasn't already valid OR was classified as a depth-discontinuity
    edge by :func:`plumbline.metrics.masks.boundary_edge_mask`.
    """
    from plumbline.metrics.masks import boundary_edge_mask

    gt_arr = np.asarray(gt)
    if valid is None:
        base_valid = np.isfinite(gt_arr) & (gt_arr > 0)
    else:
        base_valid = np.asarray(valid, dtype=bool) & np.isfinite(gt_arr) & (gt_arr > 0)

    if gt_arr.ndim == 2:
        edge = boundary_edge_mask(gt_arr, base_valid, thickness=thickness, tol=tol)
        return base_valid & ~edge

    # Batched (N, H, W): process each view independently — the boundary
    # method is per-image.
    out = np.empty_like(base_valid)
    for i in range(gt_arr.shape[0]):
        edge_i = boundary_edge_mask(gt_arr[i], base_valid[i], thickness=thickness, tol=tol)
        out[i] = base_valid[i] & ~edge_i
    return out


def _back_project_depth(
    depth: NDArray[Any], intrinsics: NDArray[Any], extrinsics: NDArray[Any]
) -> NDArray[np.float32]:
    """Lift ``(N, H, W)`` depth + ``(N, 3, 3)`` K + ``(N, 4, 4)`` E_world_from_cam
    into a ``(N, H, W, 3)`` point map in the world frame.

    Matches the implicit point map that VGGT-style adapters return directly:
    per-pixel (u, v, d) → camera-frame (x_cam, y_cam, z_cam) via K^-1,
    then world-frame via the extrinsic. Invalid pixels (depth ≤ 0 or
    non-finite) become zero points. Vectorised over views.

    Used by :func:`_compute_metrics` to unlock chamfer/F-score for
    adapters (DA3, DA-V2 metric) that return depth but not a dense
    point_map directly.
    """
    depth = np.asarray(depth)
    K = np.asarray(intrinsics, dtype=np.float64)
    E = np.asarray(extrinsics, dtype=np.float64)
    N, H, W = depth.shape
    # Pixel grid, same for every view.
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (H, W)
    out = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        fx = K[i, 0, 0]
        fy = K[i, 1, 1]
        cx = K[i, 0, 2]
        cy = K[i, 1, 2]
        d = np.asarray(depth[i], dtype=np.float64)
        valid = np.isfinite(d) & (d > 0)
        x_cam = (uu - cx) * d / max(fx, 1e-12)
        y_cam = (vv - cy) * d / max(fy, 1e-12)
        z_cam = d
        # (H, W, 3) in cam frame, zero where invalid.
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        p_cam = np.where(valid[..., None], p_cam, 0.0)
        # World = E @ [x; y; z; 1]; E is world_from_camera.
        R = E[i, :3, :3]
        t = E[i, :3, 3]
        p_world = p_cam @ R.T + t  # (H, W, 3)
        # Keep invalid pixels as exact zeros (chamfer skips them when
        # masked, though currently the metric runs on all points).
        p_world = np.where(valid[..., None], p_world, 0.0)
        out[i] = p_world.astype(np.float32)
    return out


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
