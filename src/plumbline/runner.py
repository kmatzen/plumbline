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
    voxel_downsample,
)
from plumbline.metrics.pose import accuracy_at_threshold as pose_acc_fn
from plumbline.metrics.pose import auc as pose_auc_fn
from plumbline.metrics.pose import (
    pairwise_pose_errors,
    rotation_error_degrees,
    trajectory_ate_rmse_sim3,
    trajectory_rpe_rmse_sim3,
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
    pose_acc_thresholds: tuple[float, ...] = (15.0,),
    pose_auc_mode: str = "analytic",
    pose_translation_antipodal: bool = False,
    pose_trajectory_metrics: bool = False,
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
    per_view_masked: bool = False,
    per_view_crop: int = 224,
    geometric_consistency: bool = False,
    geo_pixel_thres: float = 1.0,
    geo_depth_thres: float = 0.01,
    geo_mask_thres: int = 3,
    geo_num_src_views: int = 4,
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
    # Per-view-masked path (CUT3R/MASt3R/VGGT-family DTU eval): we
    # additionally accumulate per-view-masked GT pts3d alongside the
    # masked predictions. ICP + chamfer at scene-merge time then run
    # against the accumulated GT, NOT against sample.point_cloud_gt.
    scene_gt_masked: dict[str, list[NDArray[np.float32]]] = {}

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
        if (
            aggregation == "scene"
            and per_view_masked
            and trimmed_sample.depth_gt is not None
            and trimmed_sample.depth_valid is not None
        ):
            # Per-view-masked chamfer (CUT3R/MASt3R/VGGT DTU lineage):
            # build per-view masked pred+GT pts3d clouds at the model's
            # processed resolution, center-crop to ``per_view_crop``,
            # mask by GT validity, and accumulate per scene. Scene-merge
            # below does ICP + KDTree NN against the accumulated GT, NOT
            # against ``sample.point_cloud_gt``. Reference:
            # ``CUT3R/eval/mv_recon/launch.py`` lines ~195-260 +
            # ``utils.py::accuracy/completion``.
            geo_mask = None
            if geometric_consistency:
                geo_mask = _geometric_consistency_mask(
                    prediction=prediction,
                    num_src_views=geo_num_src_views,
                    geo_pixel_thres=geo_pixel_thres,
                    geo_depth_thres=geo_depth_thres,
                    geo_mask_thres=geo_mask_thres,
                )
            pv = _per_view_masked_clouds(
                prediction=prediction,
                sample=trimmed_sample,
                per_view_crop=per_view_crop,
                geo_mask=geo_mask,
            )
            if pv is not None:
                pred_pts, gt_pts = pv
                # Per-chunk voxel_downsample BEFORE accumulating, mirroring
                # the legacy scene-merged path. ETH3D clouds (millions of
                # points / scene) make scene-agg ICP + chamfer untractable
                # without this; DTU clouds (~800 K / scan) don't strictly
                # need it. CAUTION: this voxel applies to pred AND gt at
                # their per-sample frames — if pred and gt are in different
                # units (VGGT-DTU: pred ≈ metres, gt = mm), a single
                # ``scene_voxel_size`` would collapse one side. The cleanest
                # escape is ``scene_voxel_size: 0`` (DTU's choice) and let
                # the scene-level ICP + chamfer absorb the larger clouds.
                # ETH3D pred and gt happen to share metres so it works.
                if scene_voxel_size and scene_voxel_size > 0:
                    pred_pts = voxel_downsample(pred_pts, scene_voxel_size).astype(np.float32)
                    gt_pts = voxel_downsample(gt_pts, scene_voxel_size).astype(np.float32)
                scene = sample.sample_id.split("/", 1)[0]
                scene_points.setdefault(scene, []).append(pred_pts)
                scene_gt_masked.setdefault(scene, []).append(gt_pts)
                report.per_sample.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        metrics={"n_pred_points": float(pred_pts.shape[0])},
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
                        skip_reason="per-view-masked: missing pred point_map/depth or GT depth",
                    )
                )
            continue

        if aggregation == "scene" and trimmed_sample.point_cloud_gt is not None:
            # Per-sample alignment in scene mode is a cheap warm-start only:
            # camera_centers Umeyama puts each chunk in roughly the right
            # neighbourhood of the GT frame. Any user-requested ICP refine
            # is deferred to a single scene-level pass below, on the fused +
            # voxel-downsampled cloud. Running ICP per sample against a
            # 200 K-point GT scan was O(N_samples × 30 iters × KDTree query)
            # and dominated the rental-run wall time on ETH3D + DTU.
            per_sample_align = (
                "camera_centers" if pointcloud_alignment == "icp" else pointcloud_alignment
            )
            aligned = _aligned_point_map(
                prediction=prediction,
                sample=trimmed_sample,
                pointcloud_alignment=per_sample_align,
            )
            if aligned is not None:
                scene = sample.sample_id.split("/", 1)[0]
                # Voxel-downsample each per-sample chunk BEFORE accumulating.
                # Without this, DTU's 22 × ~42 predictions × ~1 M points each
                # balloon to 10-20 GB of in-memory chunks and the aggregation
                # OOM-kills on boxes with <32 GB RAM (D20 in DISCREPANCIES,
                # exit 137 observed 2026-04-23). The fused-cloud downsample at
                # the merge step (line ~340) was doing exactly this per-voxel
                # unification already; moving it earlier just bounds peak RSS
                # to ~O(N_scenes × scene_voxels) with no algorithmic change.
                chunk_pts = aligned.reshape(-1, 3)
                if chunk_pts.shape[0] > 0:
                    chunk_pts = voxel_downsample(chunk_pts, scene_voxel_size).astype(np.float32)
                scene_points.setdefault(scene, []).append(chunk_pts)
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
            pose_acc_thresholds=pose_acc_thresholds,
            pose_auc_mode=pose_auc_mode,
            pose_translation_antipodal=pose_translation_antipodal,
            pose_trajectory_metrics=pose_trajectory_metrics,
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
        scene_task = scene_progress.add_task("chamfer", total=len(scene_points), status="")
        scene_progress.start()
        per_scene: dict[str, dict[str, float]] = {}
        for scene, chunks in scene_points.items():
            scene_progress.update(scene_task, status=f"scene={scene}")
            merged = np.vstack(chunks).astype(np.float32)
            if per_view_masked and scene in scene_gt_masked:
                # Per-view-masked path: GT is the accumulated per-view
                # masked GT pts3d (in rebased GT world frame). ICP align
                # pred → GT, then KDTree NN both directions on the raw
                # masked clouds (no extra voxel_downsample).
                gt = np.vstack(scene_gt_masked[scene]).astype(np.float32)
                if pointcloud_alignment == "icp" and merged.shape[0] >= 3 and gt.shape[0] >= 3:
                    # Warm-start with the cloud-bbox-centroid + bbox-
                    # diagonal scale ratio so ICP doesn't collapse to s=0
                    # when pred and GT are in different units (VGGT
                    # outputs ≈ metres while DTU GT is in mm). Without a
                    # warm start, the first Umeyama with mostly-noise
                    # correspondences picks s ≈ 0 and the iterations are
                    # stuck. Bbox-diagonal is the simplest similarity
                    # match that's symmetric and unit-aware.
                    bbox_s, bbox_R, bbox_t = _bbox_similarity_warm_start(merged, gt)
                    s, R, t, _info = icp_similarity(
                        merged, gt, init_s=bbox_s, init_R=bbox_R, init_t=bbox_t
                    )
                    merged = apply_similarity(merged, s, R, t).astype(np.float32)
                per_scene[scene] = accuracy_completeness(
                    merged,
                    gt,
                    voxel_size=None,
                    outlier_distance=chamfer_outlier_distance,
                )
                scene_progress.advance(scene_task, 1)
                continue
            gt = scene_gt[scene]
            if pointcloud_alignment == "icp":
                # Single scene-level ICP refine, on the fused + voxel-
                # downsampled prediction cloud. Camera-centres Umeyama
                # already put each per-sample chunk in the right neighbourhood
                # above; this tightens the global fit against the GT scan.
                merged_ds = voxel_downsample(merged, scene_voxel_size).astype(np.float32)
                if merged_ds.shape[0] >= 3 and gt.shape[0] >= 3:
                    s, R, t, _info = icp_similarity(merged_ds, gt)
                    merged = apply_similarity(merged, s, R, t).astype(np.float32)
            # Per-chunk voxel_downsample already ran in the per-sample loop
            # at scene_voxel_size resolution. Skip the second downsample
            # inside ``accuracy_completeness`` — CUT3R/MASt3R/VGGT-family
            # eval (CUT3R's ``eval/mv_recon/utils.py``) computes Acc/Comp
            # as raw KDTree NN distances on the masked cloud, no
            # additional downsample. A re-downsample of an already-
            # voxel-downsampled cloud unifies per-chunk grids by
            # averaging centroids, which tends to push points slightly
            # further from any individual surface measurement and inflate
            # Acc relative to the paper convention.
            per_scene[scene] = accuracy_completeness(
                merged,
                gt,
                voxel_size=None,
                outlier_distance=chamfer_outlier_distance,
            )
            scene_progress.advance(scene_task, 1)
        scene_progress.stop()
        report.per_scene_metrics = per_scene
        keys = sorted({k for m in per_scene.values() for k in m})
        # Aggregate across scenes with an unweighted mean, skipping scenes
        # whose value is non-finite — mirrors the per-sample path below
        # ("skip NaNs"). A single scene whose scene-merge ICP/chamfer
        # diverged to NaN must not poison the whole benchmark number:
        # observed on pi3-DTU, where one divergent scan turned the 22-scan
        # Overall into NaN while the converged scenes were fine (ETH3D, with
        # the same per-view-masked path but fewer scenes, stayed finite).
        scene_aggregate: dict[str, float] = {}
        for k in keys:
            vals = np.asarray(
                [per_scene[s][k] for s in per_scene if k in per_scene[s]],
                dtype=np.float64,
            )
            finite = vals[np.isfinite(vals)]
            scene_aggregate[k] = float(finite.mean()) if finite.size else float("nan")
        report.aggregate_metrics = scene_aggregate
        primary = "overall" if "overall" in keys else (keys[0] if keys else None)
        if primary is not None:
            dropped = [
                s
                for s in per_scene
                if primary in per_scene[s] and not np.isfinite(per_scene[s][primary])
            ]
            if dropped:
                log.warning(
                    "scene-agg '%s': dropped %d/%d non-finite scene(s) before "
                    "aggregation; per-scene=%s",
                    primary,
                    len(dropped),
                    len(per_scene),
                    {s: per_scene[s].get(primary) for s in per_scene},
                )
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


def _sample_input_fingerprint(sample: Sample) -> str:
    """Short stable hash of the loader's actual output for this sample.

    Mixed into the prediction-cache key so a loader refactor that changes
    preprocessing (resolution, cropping, color conversion, etc.) auto-
    invalidates the cache. Before this, the cache key was just
    ``(model, model_config, dataset_name, sample_id)`` — a loader change
    that kept ``sample_id`` stable but altered the tensor shape would
    serve stale predictions against fresh GT and silently produce wrong
    metrics (D21 in docs/DISCREPANCIES.md, observed 2026-04-24 on the
    MoGe-KITTI warp fix where a pre-warp 1242×375 prediction was scored
    against a post-warp 750×375 GT).

    The fingerprint is a sha1 over the full image shape + dtype + the
    intrinsics + the first 1 KB of image bytes. O(μs) per sample and
    ≤8 hex chars added to the cache filename.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(str(sample.images.shape).encode())
    h.update(str(sample.images.dtype).encode())
    h.update(sample.intrinsics.tobytes())
    # A few KB of raw pixel bytes catches subtler changes (resampling
    # kernel, colour-space flip) that shape + intrinsics alone miss.
    h.update(sample.images.tobytes()[:1024])
    return h.hexdigest()[:8]


def _predict_with_cache(
    *,
    model: Model,
    dataset_name: str,
    sample: Sample,
    max_views: int,
    cache: PredictionCache,
) -> Prediction | None:
    fingerprint = _sample_input_fingerprint(sample)
    cache_args = (
        model.name,
        model.config_hash(),
        dataset_name,
        sample.sample_id,
    )
    if cache.has(*cache_args, input_fingerprint=fingerprint):
        try:
            return cache.load(*cache_args, input_fingerprint=fingerprint)
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
        cache.save(*cache_args, prediction=prediction, input_fingerprint=fingerprint)
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
    pose_acc_thresholds: tuple[float, ...],
    pose_auc_mode: str,
    pose_translation_antipodal: bool,
    pose_trajectory_metrics: bool,
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
        out.update(
            _pose_metrics(
                prediction.extrinsics,
                sample.extrinsics_gt,
                pose_auc_thresholds,
                pose_acc_thresholds,
                pose_auc_mode,
                pose_translation_antipodal,
                pose_trajectory_metrics,
            )
        )

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
        if sample.point_cloud_gt is None:
            log.warning("pointcloud_alignment=icp needs sample.point_cloud_gt; leaving unaligned")
        else:
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


def _bbox_similarity_warm_start(
    src: NDArray[Any], dst: NDArray[Any]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Coarse Sim(3) warm start matching bbox centre + diagonal ratio.

    For ICP between clouds in different units (e.g. VGGT predicts in
    metres, DTU GT is in mm), the per-iteration Umeyama-on-noisy-NN-
    correspondences collapses to s≈0 unless seeded with a sensible
    scale. Bbox-diagonal ratio is unit-aware and rotation-free, which
    is enough to bring the two clouds into the same neighbourhood.
    Returns (s, R=I, t).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return 1.0, np.eye(3), np.zeros(3)
    src_min, src_max = src.min(axis=0), src.max(axis=0)
    dst_min, dst_max = dst.min(axis=0), dst.max(axis=0)
    src_diag = float(np.linalg.norm(src_max - src_min))
    dst_diag = float(np.linalg.norm(dst_max - dst_min))
    s = dst_diag / max(src_diag, 1e-12)
    src_mid = 0.5 * (src_min + src_max)
    dst_mid = 0.5 * (dst_min + dst_max)
    t = dst_mid - s * src_mid
    return s, np.eye(3), t


def _geometric_consistency_mask(
    *,
    prediction: Prediction,
    num_src_views: int,
    geo_pixel_thres: float,
    geo_depth_thres: float,
    geo_mask_thres: int,
) -> NDArray[np.bool_] | None:
    """Per-view geometric-consistency mask, ported from PatchmatchNet's eval.

    For each ref view i, reproject ref→src→ref through every pred-cam pair
    using ref's depth and src's depth (top-K nearest src views by camera
    centre). A ref pixel agrees with a src view when reproj-pixel-error
    < ``geo_pixel_thres`` AND relative-depth-diff < ``geo_depth_thres``.
    Pixels that agree with at least ``geo_mask_thres`` source views are
    kept.

    This is the post-processing cited by MASt3R §4.5 ("we remove spurious
    3D points via geometric consistency post-processing [99]") and inherited
    by VGGT §4.2 ("Following MASt3R"). Reference [99] is Wang et al.,
    PatchmatchNet, CVPR 2021. Verbatim port of
    ``FangjinhuaWang/PatchmatchNet`` ``eval.py::reproject_with_depth`` +
    ``check_geometric_consistency`` + ``filter_depth`` (lines 86-255).

    Convention adapter: PatchmatchNet stores extrinsics as
    ``cam_from_world``; plumbline as ``world_from_cam``. Substitute
    ``E_src_cam_from_world @ E_world_from_ref_cam`` →
    ``inv(Ew_src) @ Ew_ref``.

    Returns a boolean ``(V, H, W)`` array, or ``None`` if essential
    prediction fields are missing.
    """
    if prediction.depth is None or prediction.intrinsics is None or prediction.extrinsics is None:
        return None
    depth = prediction.depth.astype(np.float64)  # (V, H, W) — predicted z-depth
    K = prediction.intrinsics.astype(np.float64)  # (V, 3, 3) at pred-pixel res
    Ew = prediction.extrinsics.astype(np.float64)  # (V, 4, 4) world_from_cam
    if depth.ndim != 3 or K.shape[0] != depth.shape[0] or Ew.shape[0] != depth.shape[0]:
        return None
    V, H, W = depth.shape
    if V < 2 or num_src_views < 1:
        return np.ones((V, H, W), dtype=np.bool_)

    # Camera centres in world frame: c_w = Ew @ [0,0,0,1] = Ew[:,3].
    centres = Ew[:, :3, 3]  # (V, 3)
    # Pairwise distances; for each ref pick top-K closest other views.
    # PatchmatchNet uses ``pair.txt`` (visual-overlap ranking from
    # MVSNet preprocessing); we approximate with camera-centre k-NN.
    D = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    src_order = np.argsort(D, axis=1)[:, : int(num_src_views)]  # (V, k)

    # Pre-build pixel grid once; reused for every (ref, src) pair.
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    x_ref_flat = x_grid.reshape(-1).astype(np.float64)
    y_ref_flat = y_grid.reshape(-1).astype(np.float64)
    ones_flat = np.ones_like(x_ref_flat)

    geo_sum = np.zeros((V, H, W), dtype=np.int32)
    eps = 1e-12
    for i in range(V):
        K_ref = K[i]
        K_ref_inv = np.linalg.inv(K_ref)
        Ew_ref_inv = np.linalg.inv(Ew[i])  # cam_from_world for ref
        d_ref = depth[i].reshape(-1)
        valid_ref = d_ref > 0
        if not valid_ref.any():
            continue
        # Ref pixels → ref-cam 3D
        xyz_ref_cam = K_ref_inv @ np.vstack((x_ref_flat, y_ref_flat, ones_flat)) * d_ref
        for j in src_order[i]:
            K_src = K[j]
            d_src = depth[j]
            # cam_from_world @ world_from_ref_cam = src_cam_from_ref_cam
            T_src_from_ref = np.linalg.inv(Ew[j]) @ Ew[i]
            xyz_in_src = T_src_from_ref[:3, :3] @ xyz_ref_cam + T_src_from_ref[:3, 3:4]
            uv_src_h = K_src @ xyz_in_src
            z_src_proj = uv_src_h[2]
            # Avoid div-by-zero at the rear half-space.
            valid_proj = z_src_proj > eps
            xy_src = np.where(
                valid_proj, uv_src_h[:2] / np.where(valid_proj, z_src_proj, 1.0), -1.0
            )
            x_src = xy_src[0]
            y_src = xy_src[1]
            # Bilinear-ish: nearest-pixel sample (matches PatchmatchNet's
            # cv2.remap default but using NN here for simplicity / no
            # cv2 dependency; bilinear is straightforward to add later).
            xi = np.round(x_src).astype(np.int64)
            yi = np.round(y_src).astype(np.int64)
            in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid_proj
            sampled = np.zeros_like(d_ref)
            sampled[in_bounds] = d_src[yi[in_bounds], xi[in_bounds]]
            # Reproject src → ref-cam → ref-pixel
            xyz_src_cam = np.linalg.inv(K_src) @ np.vstack((xy_src, ones_flat)) * sampled
            T_ref_from_src = Ew_ref_inv @ Ew[j]
            xyz_back_ref_cam = T_ref_from_src[:3, :3] @ xyz_src_cam + T_ref_from_src[:3, 3:4]
            depth_reproj = xyz_back_ref_cam[2]
            uv_back = K_ref @ xyz_back_ref_cam
            z_back = uv_back[2]
            valid_back = (z_back > eps) & in_bounds & (sampled > 0)
            xy_back = np.where(valid_back, uv_back[:2] / np.where(valid_back, z_back, 1.0), 1e9)
            dist_px = np.sqrt((xy_back[0] - x_ref_flat) ** 2 + (xy_back[1] - y_ref_flat) ** 2)
            rel_depth_diff = np.where(
                d_ref > eps, np.abs(depth_reproj - d_ref) / np.maximum(d_ref, eps), 1e9
            )
            agree = (
                (dist_px < geo_pixel_thres)
                & (rel_depth_diff < geo_depth_thres)
                & valid_back
                & valid_ref
            )
            geo_sum[i] += agree.reshape(H, W).astype(np.int32)

    return geo_sum >= int(geo_mask_thres)


def _per_view_masked_clouds(
    *,
    prediction: Prediction,
    sample: Sample,
    per_view_crop: int,
    geo_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """Build per-view 3D clouds for the CUT3R per-view-masked chamfer.

    For each of the ``V`` views:

    - Pred pts3d come from ``prediction.point_map`` (already world-frame)
      or by lifting ``prediction.depth`` with ``prediction.intrinsics`` +
      ``prediction.extrinsics``.
    - GT depth is NN-downsampled to the prediction's processed resolution
      (``H_p, W_p``); GT pts3d are unprojected with sample's intrinsics
      *rescaled to (H_p, W_p)* and ``sample.extrinsics_gt`` (rebased to
      view 0).
    - A per_view_crop x per_view_crop center crop is taken on both;
      pixels failing the GT validity mask are dropped.

    Returns flat ``(M, 3)`` pred + GT clouds, OR ``None`` when essential
    data is missing.

    The two clouds end up in *different* world frames (pred world vs
    rebased-GT world) — the scene-merge step's ICP aligns them.
    """
    if sample.depth_gt is None or sample.depth_valid is None:
        return None
    pmap = prediction.point_map
    if pmap is None:
        if (
            prediction.depth is None
            or prediction.intrinsics is None
            or prediction.extrinsics is None
        ):
            return None
        pmap = _back_project_depth(prediction.depth, prediction.intrinsics, prediction.extrinsics)
    V_p, H_p, W_p, _ = pmap.shape
    V_gt, H_gt_canvas, W_gt_canvas = sample.depth_gt.shape
    if V_p != V_gt:
        log.warning("per-view-masked: pred view count %d != GT view count %d", V_p, V_gt)
        return None

    # Per-view native sizes — datasets with varying-native-size views (ETH3D)
    # populate ``metadata['native_sizes']`` as a list of (H_i, W_i) at *image*
    # native; datasets with uniform views (DTU) leave it unset and the GT
    # canvas IS native. ``metadata['gt_sizes']`` separately describes the
    # depth_gt's per-view actual extent within the canvas, useful when the
    # loader renders at a smaller-than-native resolution to keep cache size
    # bounded (e.g. ETH3D rendered at max-dim 2048 vs 6048 image native).
    # When unset, gt_sizes defaults to native_sizes.
    native_sizes = sample.metadata.get("native_sizes") if sample.metadata else None
    if native_sizes is None:
        native_sizes = [(H_gt_canvas, W_gt_canvas)] * V_p
    gt_sizes = sample.metadata.get("gt_sizes") if sample.metadata else None
    if gt_sizes is None:
        gt_sizes = native_sizes

    K_native = sample.intrinsics.astype(np.float64)  # (V, 3, 3) in per-view native px
    E_world_from_cam = sample.extrinsics_gt.astype(np.float64)  # (V, 4, 4) rebased
    u = np.arange(W_p, dtype=np.float64)
    v = np.arange(H_p, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    gt_pts_world = np.empty((V_p, H_p, W_p, 3), dtype=np.float64)
    gt_depth_p = np.empty((V_p, H_p, W_p), dtype=np.float64)
    gt_valid_p = np.zeros((V_p, H_p, W_p), dtype=np.bool_)
    for i in range(V_p):
        H_n, W_n = int(native_sizes[i][0]), int(native_sizes[i][1])
        H_g, W_g = int(gt_sizes[i][0]), int(gt_sizes[i][1])
        # NN-sample GT depth at the depth_gt-render resolution. depth_gt is
        # stored at canvas size with the actual data in [..., :H_g, :W_g];
        # the rest is padding (zero depth / False valid).
        yi = (np.arange(H_p) * H_g / H_p).astype(np.int64)
        xi = (np.arange(W_p) * W_g / W_p).astype(np.int64)
        d = sample.depth_gt[i, yi[:, None], xi[None, :]].astype(np.float64)
        valid = sample.depth_valid[i, yi[:, None], xi[None, :]] & (d > 0)
        # Per-view K rescaled image-native → pred res. K_native is in image
        # native px (matches sample.images for view i); pred_pixel
        # corresponds to image_native_pixel × (H_p/H_n, W_p/W_n) by
        # construction of the model adapter's resize.
        sx_i = W_p / max(W_n, 1)
        sy_i = H_p / max(H_n, 1)
        fx = K_native[i, 0, 0] * sx_i
        fy = K_native[i, 1, 1] * sy_i
        cx = K_native[i, 0, 2] * sx_i
        cy = K_native[i, 1, 2] * sy_i
        x_cam = (uu - cx) * d / max(fx, 1e-12)
        y_cam = (vv - cy) * d / max(fy, 1e-12)
        z_cam = d
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        R = E_world_from_cam[i, :3, :3]
        t = E_world_from_cam[i, :3, 3]
        gt_pts_world[i] = p_cam @ R.T + t
        gt_depth_p[i] = d
        gt_valid_p[i] = valid

    # Center crop both pred + GT to per_view_crop x per_view_crop. Crop is
    # in pred-pixel coordinates (where the model "sees" 224x224 of detail);
    # if the prediction is smaller than the crop, the crop falls back to
    # the full image with a warning so the run continues.
    side = int(per_view_crop)
    if side > min(H_p, W_p):
        log.warning(
            "per_view_crop %d larger than processed resolution (%d, %d); using full image",
            side,
            H_p,
            W_p,
        )
        crop = (slice(None), slice(None))
    else:
        cy_p = H_p // 2
        cx_p = W_p // 2
        half = side // 2
        crop = (slice(cy_p - half, cy_p + half), slice(cx_p - half, cx_p + half))

    pred_crop = pmap[:, crop[0], crop[1], :]
    gt_crop = gt_pts_world[:, crop[0], crop[1], :]
    valid_crop = gt_valid_p[:, crop[0], crop[1]]

    # AND in the optional geometric-consistency mask (PatchmatchNet [99],
    # cited by MASt3R §4.5 and inherited by VGGT §4.2 via "Following MASt3R").
    # Mask is at pred-pixel resolution; crop the same window.
    if geo_mask is not None:
        if geo_mask.shape != pmap.shape[:3]:
            log.warning(
                "geo_mask shape %s != pred (V,H,W) %s; skipping",
                geo_mask.shape,
                pmap.shape[:3],
            )
        else:
            valid_crop = valid_crop & geo_mask[:, crop[0], crop[1]]

    pred_concat = pred_crop[valid_crop].astype(np.float32, copy=False)
    gt_concat = gt_crop[valid_crop].astype(np.float32, copy=False)
    if pred_concat.shape[0] == 0 or gt_concat.shape[0] == 0:
        return None
    return pred_concat, gt_concat


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
    acc_thresholds: tuple[float, ...],
    auc_mode: str,
    translation_antipodal: bool,
    trajectory_metrics: bool = False,
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
    abs_trans = np.asarray(
        translation_cosine_error(Ep[..., :3, 3], Eg[..., :3, 3], antipodal=translation_antipodal)
    ).reshape(-1)
    abs_combined = np.maximum(abs_rot, abs_trans)
    abs_aucs = pose_auc_fn(abs_combined, list(auc_thresholds), mode=auc_mode)

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
        pw_rot, pw_trans = pairwise_pose_errors(
            E_pred, E_gt, translation_antipodal=translation_antipodal
        )
        pw_combined = np.maximum(pw_rot, pw_trans)
        pw_aucs = pose_auc_fn(pw_combined, list(auc_thresholds), mode=auc_mode)
        pw_rra = pose_acc_fn(pw_rot, list(acc_thresholds))
        pw_rta = pose_acc_fn(pw_trans, list(acc_thresholds))
        out["pairwise_rot_err_deg_mean"] = float(np.nanmean(pw_rot))
        out["pairwise_trans_cos_err_deg_mean"] = float(np.nanmean(pw_trans))
        for t, v in pw_aucs.items():
            out[f"pairwise_pose_auc@{t:g}"] = float(v)
        for t, v in pw_rra.items():
            out[f"pairwise_RRA@{t:g}"] = float(v)
        for t, v in pw_rta.items():
            out[f"pairwise_RTA@{t:g}"] = float(v)

    # Trajectory pose metrics (TUM-RGBD-style ATE / RPE). These are what
    # MonST3R Table 4 / SLAM-style papers report for video-pose evaluation.
    # The metric is meaningful only when each Sample is a *trajectory* (N
    # ordered frames from one sequence) AND the loader provides
    # metric-scale GT (Sim(3) alignment recovers scale, so any
    # consistently-scaled pred lands on the same number). Plumbline
    # delegates to `evo` to match MonST3R's bit-for-bit eval shape (their
    # `dust3r/utils/vo_eval.py:eval_metrics` is the same evo wrapper).
    if E_pred.ndim == 3 and E_pred.shape[0] >= 3 and trajectory_metrics:
        try:
            ate = trajectory_ate_rmse_sim3(E_pred, E_gt)
            rpe_t, rpe_r = trajectory_rpe_rmse_sim3(E_pred, E_gt, delta=1)
            out["trajectory_ate_rmse"] = float(ate)
            out["trajectory_rpe_trans_rmse"] = float(rpe_t)
            out["trajectory_rpe_rot_deg_rmse"] = float(rpe_r)
        except ImportError:
            # `evo` is an optional extra; silently skip if not installed.
            pass
        except Exception:
            # Umeyama can fail on degenerate trajectories (collinear /
            # near-stationary cameras). Emit NaN so the per-sample row
            # records the failure and aggregation skips it via nanmean —
            # don't crash the whole eval.
            out["trajectory_ate_rmse"] = float("nan")
            out["trajectory_rpe_trans_rmse"] = float("nan")
            out["trajectory_rpe_rot_deg_rmse"] = float("nan")

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
