"""Metric implementations: pure numpy, side-effect-free."""

from plumbline.metrics.alignment import (
    align_scale_and_shift,
    align_scale_lstsq,
    align_scale_median,
)
from plumbline.metrics.depth import abs_rel, delta_threshold, rmse, silog
from plumbline.metrics.pointmap import chamfer_distance, f_score
from plumbline.metrics.pose import (
    auc,
    pose_auc,
    rotation_error_degrees,
    translation_cosine_error,
    translation_error,
)

__all__ = [
    "abs_rel",
    "align_scale_and_shift",
    "align_scale_lstsq",
    "align_scale_median",
    "auc",
    "chamfer_distance",
    "delta_threshold",
    "f_score",
    "pose_auc",
    "rmse",
    "rotation_error_degrees",
    "silog",
    "translation_cosine_error",
    "translation_error",
]
