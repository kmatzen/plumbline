"""Metric implementations: pure numpy, side-effect-free."""

from plumbline.metrics.alignment import (
    align_scale_and_shift,
    align_scale_lstsq,
    align_scale_median,
)
from plumbline.metrics.depth import (
    abs_rel,
    delta_threshold,
    log10_error,
    rmse,
    rmse_log,
    silog,
    sq_rel,
)
from plumbline.metrics.pointmap import chamfer_distance, f_score
from plumbline.metrics.pose import (
    accuracy_at_threshold,
    auc,
    pose_auc,
    rotation_error_degrees,
    translation_cosine_error,
    translation_error,
)

__all__ = [
    "abs_rel",
    "accuracy_at_threshold",
    "align_scale_and_shift",
    "align_scale_lstsq",
    "align_scale_median",
    "auc",
    "chamfer_distance",
    "delta_threshold",
    "f_score",
    "log10_error",
    "pose_auc",
    "rmse",
    "rmse_log",
    "rotation_error_degrees",
    "silog",
    "sq_rel",
    "translation_cosine_error",
    "translation_error",
]
