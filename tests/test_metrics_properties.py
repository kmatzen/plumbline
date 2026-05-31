"""Property-based tests for depth metrics and alignment.

Unit tests pin specific values; property tests assert invariants that must
hold for *any* reasonable input. The combination catches a strictly wider
class of bugs — especially numerical edge cases that show up on real data
(all-zero regions, sky pixels with near-zero valid depth, degenerate scale
factors) but that we'd never think to write as a unit test.

Uses ``hypothesis`` to sample depth arrays, validity masks, and alignment
inputs, and asserts invariants from the metric definitions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from plumbline.metrics.alignment import (
    align_depth,
    align_scale_and_shift,
    align_scale_lstsq,
    align_scale_median,
)
from plumbline.metrics.depth import abs_rel, delta_threshold, rmse, silog

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# ``width=32`` keeps hypothesis from generating float64 values that can't
# round-trip into the float32 arrays below. Bounds are cast through
# ``float(np.float32(...))`` because newer hypothesis enforces that the
# min/max arguments are themselves exact at the requested width — 1e-4 and
# 1e3 aren't float32-exact, but their nearest-representable neighbours are.
_FINITE = st.floats(
    min_value=float(np.float32(1e-4)),
    max_value=float(np.float32(1e3)),
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# 1D arrays of positive, finite depths with modest length so hypothesis
# searches run fast.
_depth_array = arrays(
    dtype=np.float32,
    shape=st.integers(min_value=2, max_value=64),
    elements=_FINITE,
)


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------


class TestDepthMetricInvariants:
    @given(gt=_depth_array)
    def test_abs_rel_is_zero_when_pred_equals_gt(self, gt: np.ndarray) -> None:
        assert abs_rel(gt, gt) == pytest.approx(0.0, abs=1e-10)

    @given(gt=_depth_array, scale=st.floats(min_value=0.1, max_value=10))
    def test_abs_rel_is_nonnegative(self, gt: np.ndarray, scale: float) -> None:
        pred = (gt * scale).astype(np.float32)
        val = abs_rel(pred, gt)
        assert val >= 0 and math.isfinite(val)

    @given(gt=_depth_array)
    def test_rmse_is_zero_when_equal(self, gt: np.ndarray) -> None:
        assert rmse(gt, gt) == pytest.approx(0.0, abs=1e-10)

    @given(gt=_depth_array, scale=st.floats(min_value=0.1, max_value=10))
    def test_rmse_scales_linearly_with_error(self, gt: np.ndarray, scale: float) -> None:
        # For pred = scale * gt, RMSE = sqrt(mean((scale-1)^2 * gt^2)) = |scale-1| * sqrt(mean(gt^2)).
        # Tolerance is loose because `gt * scale` rounds in float32, which can
        # introduce relative error up to O(1e-3) for scales near 1.
        pred = (gt * scale).astype(np.float32)
        observed = rmse(pred, gt)
        expected = abs(scale - 1) * float(np.sqrt(np.mean(gt.astype(np.float64) ** 2)))
        # For scales extremely close to 1, ``expected`` collapses to O(1e-5)
        # and the rel=1e-3 bound evaluates to <1e-8. Widen the absolute
        # tolerance so float32 rounding of ``gt * scale`` (up to O(1e-5))
        # stays inside the combined bound.
        assert observed == pytest.approx(expected, rel=1e-3, abs=1e-4)

    @given(gt=_depth_array)
    def test_delta_is_bounded(self, gt: np.ndarray) -> None:
        # Delta threshold must lie in [0, 1] regardless of the prediction.
        rng = np.random.default_rng(0)
        pred = (rng.uniform(0.1, 10.0, size=gt.shape) * gt).astype(np.float32)
        for t in (1.25, 1.25**2, 1.25**3):
            v = delta_threshold(pred, gt, threshold=t)
            assert 0.0 <= v <= 1.0

    @given(gt=_depth_array, scale=st.floats(min_value=1e-3, max_value=1e3))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_silog_is_scale_invariant(self, gt: np.ndarray, scale: float) -> None:
        """With lambda_=1, SILog of (c*pred, gt) == SILog of (pred, gt)."""
        pred = gt.copy()
        pred_scaled = (pred * scale).astype(np.float32)
        # Need at least two unique GT values or the variance term is zero and
        # scale invariance is trivially satisfied; skip degenerate arrays.
        assume(np.unique(gt).size >= 2)
        a = silog(pred, gt)
        b = silog(pred_scaled, gt)
        assert a == pytest.approx(b, abs=1e-3)


# ---------------------------------------------------------------------------
# Scale alignment
# ---------------------------------------------------------------------------


class TestAlignmentInvariants:
    @given(gt=_depth_array, scale=st.floats(min_value=0.05, max_value=20))
    def test_median_recovers_exact_scale(self, gt: np.ndarray, scale: float) -> None:
        """pred = gt / scale ⇒ align_scale_median(pred, gt) == scale."""
        pred = (gt / scale).astype(np.float32)
        s = align_scale_median(pred, gt)
        assert s == pytest.approx(scale, rel=1e-4)

    @given(gt=_depth_array, scale=st.floats(min_value=0.05, max_value=20))
    def test_lstsq_recovers_exact_scale(self, gt: np.ndarray, scale: float) -> None:
        pred = (gt / scale).astype(np.float32)
        # align_scale_lstsq guards against a near-zero ``pred @ pred``
        # denominator (returns NaN). Skip degenerate arrays that land
        # below that guard — the recovery property is defined only when
        # the normal equation is well-posed.
        assume(
            float(np.asarray(pred, dtype=np.float64) @ np.asarray(pred, dtype=np.float64)) >= 1e-6
        )
        s = align_scale_lstsq(pred, gt)
        assert s == pytest.approx(scale, rel=1e-4)

    @given(gt=_depth_array, scale=st.floats(min_value=0.05, max_value=20))
    def test_align_depth_median_then_abs_rel_is_zero(self, gt: np.ndarray, scale: float) -> None:
        pred = (gt / scale).astype(np.float32)
        aligned = align_depth(pred, gt, mode="median").astype(np.float32)
        assert abs_rel(aligned, gt) == pytest.approx(0.0, abs=1e-4)

    @given(gt=_depth_array)
    def test_scale_shift_identity_on_perfect_input(self, gt: np.ndarray) -> None:
        """Identity fit requires ≥2 distinct values; with a single unique value,
        lstsq picks the minimum-norm solution (s, b) = (0.5, 0.5), which is
        equally valid on the evaluation set but not the identity we want.
        """
        assume(np.unique(gt).size >= 2)
        s, b = align_scale_and_shift(gt, gt, space="depth")
        assert s == pytest.approx(1.0, abs=1e-4)
        assert b == pytest.approx(0.0, abs=1e-4)

    @given(
        gt=_depth_array,
        noise_scale=st.floats(min_value=1e-3, max_value=0.2),
    )
    def test_median_robust_to_a_few_outliers(self, gt: np.ndarray, noise_scale: float) -> None:
        """Median alignment should survive a single egregious outlier."""
        assume(gt.size >= 5)
        pred = gt.copy()
        # Poison one pixel with a huge outlier ratio.
        pred[0] *= 1e6
        s = align_scale_median(pred, gt)
        # Without the outlier, s would be 1. With the outlier, median is still ~1
        # (outlier only skews a single sample out of many).
        assert s == pytest.approx(1.0, rel=0.1)
        _ = noise_scale  # unused here; kept to shape the strategy


# ---------------------------------------------------------------------------
# Pose / AUC
# ---------------------------------------------------------------------------


class TestAUCInvariants:
    @given(
        errors=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=32),
            elements=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
        ),
        threshold=st.floats(min_value=0.1, max_value=100.0),
    )
    def test_auc_in_unit_interval(self, errors: np.ndarray, threshold: float) -> None:
        from plumbline.metrics.pose import auc

        result = auc(errors, [threshold])
        assert 0.0 <= result[threshold] <= 1.0

    @given(
        n=st.integers(min_value=1, max_value=32), threshold=st.floats(min_value=0.1, max_value=10)
    )
    def test_auc_perfect_is_one(self, n: int, threshold: float) -> None:
        from plumbline.metrics.pose import auc

        result = auc(np.zeros(n), [threshold])
        assert result[threshold] == pytest.approx(1.0, abs=1e-10)

    @given(
        n=st.integers(min_value=1, max_value=32), threshold=st.floats(min_value=0.1, max_value=10)
    )
    def test_auc_all_infinite_is_zero(self, n: int, threshold: float) -> None:
        from plumbline.metrics.pose import auc

        # Infinite errors are stripped, leaving zero samples; expect nan.
        # (We test that behavior explicitly rather than changing it.)
        result = auc(np.full(n, np.inf), [threshold])
        assert math.isnan(result[threshold])
