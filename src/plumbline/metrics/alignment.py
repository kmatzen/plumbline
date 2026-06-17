"""Scale alignment modes for relative-depth predictions.

Some models predict depth up to an unknown scale (Depth Anything V2) or up to
an unknown affine transform (MiDaS, many transformer depth models). Compute
the aligning transform on ground-truth-valid pixels, then apply it to the
whole prediction before metric computation.

The runner logs which mode was used; reports display it; cached predictions
store raw (unaligned) values so the alignment can be changed without
re-running inference.

Modes
-----
- ``"none"``      — identity. Use for metric models.
- ``"median"``    — scalar ``s = median(gt / pred)`` (median-of-ratios).
                    Cheap, robust to outliers, standard for "up-to-scale" eval.
- ``"median_lineage"`` — scalar ``s = median(gt) / median(pred)`` (ratio-of-
                    medians), the dust3r-lineage (CUT3R/MonST3R/DUSt3R) eval
                    code's "median scaling". Distinct from ``"median"``; needed
                    to reproduce those papers' depth numbers exactly.
- ``"lstsq"``     — scalar ``s`` minimizing ``||s*pred - gt||_2``. Closed form.
- ``"scale_shift"`` — affine ``(s, b)`` minimizing ``||s*pred + b - gt||_2`` in
                    inverse-depth or log-depth space. Used by MiDaS-family
                    eval protocols. Operates on inverse depth by default
                    (Ranftl et al. 2020, "Towards Robust Monocular Depth
                    Estimation").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plumbline.conventions import EPS

__all__ = [
    "align_depth",
    "align_scale_and_shift",
    "align_scale_and_shift_robust",
    "align_scale_lstsq",
    "align_scale_median",
    "align_scale_ratio_of_medians",
    "align_scale_weiszfeld",
    "apply_similarity",
    "icp_similarity",
    "umeyama_similarity",
]


def align_scale_median(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None
) -> float:
    """Return the scalar ``s`` such that ``s * pred`` matches ``gt`` in median ratio.

    ``s = median(gt / pred)`` over valid pixels where both are positive.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.median(g / np.maximum(p, EPS)))


def align_scale_ratio_of_medians(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None
) -> float:
    """Return the scalar ``s = median(gt) / median(pred)`` over valid pixels.

    This is the **dust3r-lineage** per-frame "median scaling" — the exact
    estimator CUT3R / MonST3R / DUSt3R's released depth-eval code uses
    (``depth_evaluation``: ``s = median(gt) / median(pred)``), *distinct* from
    :func:`align_scale_median`'s median-of-ratios ``median(gt / pred)``. The two
    agree within a few percent on some models but diverge ~10% on others
    (CUT3R-NYU: 0.0858 vs 0.0777) — so reproducing a lineage paper number needs
    this one. Verified 2026-06-15 on the prepared NYU set: CUT3R's own pipeline
    and this estimator both give 0.0858 (paper 0.086).
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    mp = float(np.median(p))
    if abs(mp) < EPS:
        return float("nan")
    return float(np.median(g) / mp)


def align_scale_lstsq(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None
) -> float:
    """Return the scalar ``s`` minimizing ``||s*pred - gt||_2``.

    Closed form: ``s = (pred . gt) / (pred . pred)``.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0 or float(p @ p) < EPS:
        return float("nan")
    return float((p @ g) / (p @ p))


def align_scale_weiszfeld(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None = None, *, n_iter: int = 10
) -> float:
    """Robust scale-only fit via Weiszfeld IRLS — the dust3r-lineage *video*
    per-sequence scale (CUT3R `eval/video_depth --align scale`,
    ``align_with_scale=True``).

    Minimizes ``sum |s*pred - gt|`` over valid pixels: init ``s = mean(gt)/mean(pred)``,
    then ``n_iter`` rounds of ``w = 1/(|s*pred - gt| + 1e-8)`` followed by the
    weighted closed form ``s = sum(w*pred*gt) / sum(w*pred^2)``; clamped to
    ``>= 1e-3``. Distinct from ``median``/``median_lineage`` (median scalings)
    and ``lstsq`` (plain L2 scale). CUT3R Table 2 (video depth) uses this for
    Bonn/Sintel/KITTI per-sequence eval.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    mp = float(np.mean(p))
    if abs(mp) < EPS:
        return float("nan")
    s = float(np.mean(g)) / mp
    for _ in range(n_iter):
        w = 1.0 / (np.abs(s * p - g) + 1e-8)
        denom = float(np.sum(w * p * p))
        if denom < EPS:
            break
        s = float(np.sum(w * p * g)) / denom
    return max(s, 1e-3)


def align_scale_and_shift(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None = None,
    *,
    space: str = "inv_depth",
) -> tuple[float, float]:
    """Return ``(s, b)`` minimizing ``||s*pred + b - gt||_2``.

    Parameters
    ----------
    space
        ``"inv_depth"`` (default) fits in inverse-depth space, then the caller
        applies the transform to inverse predictions. ``"depth"`` fits directly
        in depth space. ``"log"`` fits in log-depth.

    Notes
    -----
    The MiDaS evaluation protocol (Ranftl et al. 2020) uses inverse-depth space
    because MiDaS predicts disparity-like quantities; matches their reported
    numbers on NYU/KITTI/Sintel.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size < 2:
        return float("nan"), float("nan")
    if space == "inv_depth":
        p = 1.0 / np.maximum(p, EPS)
        g = 1.0 / np.maximum(g, EPS)
    elif space == "log":
        p = np.log(p)
        g = np.log(g)
    elif space != "depth":
        raise ValueError(f"unknown space '{space}'; use 'depth', 'inv_depth', or 'log'")
    A = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(coef[0]), float(coef[1])


def align_scale_and_shift_robust(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None = None,
    *,
    space: str = "inv_depth",
    max_iter: int = 20,
    rel_tol: float = 1e-4,
    huber_k: float = 1.345,
) -> tuple[float, float]:
    """Robust scale + shift fit — MoGe paper's ROE protocol.

    Like :func:`align_scale_and_shift` but using iteratively-reweighted
    least-squares with Huber weights so far-outlier pixels don't dominate
    the fit. This matches MoGe's reported ROE (Robust Optimal Estimation)
    alignment and closes the systematic ~15% gap plumbline's plain LSQ
    scale_shift showed against the MoGe paper on NYU (0.0342 vs 0.0297).

    Parameters
    ----------
    pred, gt, valid, space
        Same as :func:`align_scale_and_shift`.
    max_iter
        Hard cap on IRLS iterations. Typically converges in 4-8 rounds.
    rel_tol
        Convergence threshold on the (s, b) change per iteration.
    huber_k
        Huber k (tuning constant) in units of the robust scale estimator.
        1.345 gives ~95% efficiency at the Gaussian baseline and is the
        classic robust-regression default.

    Returns ``(s, b)`` such that ``s * pred_space + b ≈ gt_space``
    in the specified ``space``. Seeds from plain-LSQ so the first weighted
    pass already operates near the plain-fit neighbourhood.
    """
    p, g, _ = _valid_pairs(pred, gt, valid)
    if p.size < 2:
        return float("nan"), float("nan")
    if space == "inv_depth":
        p = 1.0 / np.maximum(p, EPS)
        g = 1.0 / np.maximum(g, EPS)
    elif space == "log":
        p = np.log(p)
        g = np.log(g)
    elif space != "depth":
        raise ValueError(f"unknown space '{space}'; use 'depth', 'inv_depth', or 'log'")

    # Seed with plain LSQ.
    A = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    s, b = float(coef[0]), float(coef[1])

    for _ in range(max_iter):
        residuals = g - (s * p + b)
        # Robust scale: median absolute deviation, rescaled for Gaussian
        # consistency (1.4826 ≈ 1/Phi^-1(0.75)).
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = max(1.4826 * mad, EPS)
        # Huber weights: 1 for |r| <= k*sigma, else k*sigma / |r|.
        abs_r = np.abs(residuals)
        w = np.ones_like(residuals)
        far = abs_r > huber_k * sigma
        w[far] = (huber_k * sigma) / np.maximum(abs_r[far], EPS)

        # Weighted LSQ: (A^T W A) x = A^T W g.
        Aw = A * w[:, None]
        gw = g * w
        new_coef, *_ = np.linalg.lstsq(Aw, gw, rcond=None)
        new_s, new_b = float(new_coef[0]), float(new_coef[1])

        rel_change = max(
            abs(new_s - s) / max(abs(s), EPS),
            abs(new_b - b) / max(abs(b), EPS),
        )
        s, b = new_s, new_b
        if rel_change < rel_tol:
            break

    return s, b


def align_depth(
    pred: NDArray[Any],
    gt: NDArray[Any],
    valid: NDArray[Any] | None = None,
    *,
    mode: str = "median",
) -> NDArray[Any]:
    """Apply the named alignment and return the aligned prediction.

    Parameters
    ----------
    pred, gt
        Same shape; any ndim. Alignment is computed on valid pixels.
    valid
        Boolean mask of pixels to use for fitting. ``None`` = use all pixels
        where both pred and gt are positive and finite.
    mode
        One of:
          - ``"none"`` — no alignment (metric models)
          - ``"median"`` / ``"lstsq"`` — scale-only in depth space
          - ``"scale_shift"`` — scale+shift in inverse-depth (disparity)
            space. Matches MiDaS / DA-V2 / MoGe "affine-invariant
            disparity" protocol.
          - ``"scale_shift_robust"`` — same as scale_shift but IRLS +
            Huber weights (MoGe paper's ROE).
          - ``"scale_shift_clamped"`` — same LSQ as scale_shift, plus
            MoGe's per-sample disparity floor at ``1/gt.max()`` before
            inverting to depth. Use on datasets with extreme-outlier
            depths (DIODE outdoor) where plain scale_shift lets a single
            pixel dominate mean AbsRel.
          - ``"scale_shift_depth"`` — scale+shift fit in DEPTH space.
            Matches Marigold / Depth Pro / "affine-invariant depth"
            protocols. The model output is a depth value where
            ``s * pred + b = gt_depth`` is the natural fit.
    """
    if mode == "none":
        return pred
    out = pred.astype(np.float64, copy=True)
    if mode == "median":
        s = align_scale_median(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "median_lineage":
        # dust3r-lineage (CUT3R/MonST3R/DUSt3R) eval code's "median scaling":
        # s = median(gt)/median(pred), NOT median-of-ratios. See
        # align_scale_ratio_of_medians.
        s = align_scale_ratio_of_medians(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "scale_weiszfeld":
        # dust3r-lineage VIDEO per-sequence scale (CUT3R --align scale):
        # robust scale-only Weiszfeld IRLS. See align_scale_weiszfeld.
        s = align_scale_weiszfeld(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "lstsq":
        s = align_scale_lstsq(pred, gt, valid)
        if np.isfinite(s):
            out *= s
        return out
    if mode == "scale_shift":
        s, b = align_scale_and_shift(pred, gt, valid, space="inv_depth")
        if np.isfinite(s) and np.isfinite(b):
            inv = 1.0 / np.maximum(out, EPS)
            inv = s * inv + b
            out = 1.0 / np.maximum(inv, EPS)
        return out
    if mode == "scale_shift_robust":
        s, b = align_scale_and_shift_robust(pred, gt, valid, space="inv_depth")
        if np.isfinite(s) and np.isfinite(b):
            inv = 1.0 / np.maximum(out, EPS)
            inv = s * inv + b
            out = 1.0 / np.maximum(inv, EPS)
        return out
    if mode == "scale_shift_clamped":
        # MoGe's disparity-space eval protocol (`moge/test/metrics.py`
        # ~line 210): same LSQ fit as ``scale_shift``, but after
        # ``s * inv + b``, clamp the minimum aligned disparity at
        # ``1 / gt_depth[mask].max()`` before inverting to depth. Bounds
        # the per-sample aligned depth above by the largest valid GT
        # depth — without this, a tiny or negative post-fit disparity
        # inverts to an enormous depth and a single pixel dominates mean
        # AbsRel. On DIODE, where outdoor returns legitimately reach
        # hundreds of metres, this is what keeps the mean from diverging
        # even while the median is bullseye-on-paper.
        s, b = align_scale_and_shift(pred, gt, valid, space="inv_depth")
        if np.isfinite(s) and np.isfinite(b):
            inv = 1.0 / np.maximum(out, EPS)
            inv = s * inv + b
            _, gt_masked, _ = _valid_pairs(pred, gt, valid)
            if gt_masked.size:
                disp_floor = 1.0 / float(gt_masked.max())
                inv = np.maximum(inv, disp_floor)
            out = 1.0 / np.maximum(inv, EPS)
        return out
    if mode == "scale_shift_depth":
        # Fit in depth space: gt_depth ≈ s * pred + b (no 1/. wrapping).
        # Matches Marigold / Depth Pro protocol where the model already
        # outputs a depth-like [0, 1] and the paper eval is depth-space
        # LSQ.
        s, b = align_scale_and_shift(pred, gt, valid, space="depth")
        if np.isfinite(s) and np.isfinite(b):
            out = s * out + b
        return out
    raise ValueError(f"unknown alignment mode '{mode}'")


def _valid_pairs(
    pred: NDArray[Any], gt: NDArray[Any], valid: NDArray[Any] | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    mask = (
        (np.ones(pred.shape, dtype=bool) if valid is None else valid)
        & np.isfinite(pred)
        & np.isfinite(gt)
        & (pred > 0)
        & (gt > 0)
    )
    return pred[mask].astype(np.float64), gt[mask].astype(np.float64), mask


# ---------------------------------------------------------------------------
# 7-DoF similarity for point clouds (ETH3D / T&T / DTU style chamfer)
# ---------------------------------------------------------------------------


def umeyama_similarity(
    src: NDArray[Any], dst: NDArray[Any]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Closed-form least-squares similarity ``src → dst`` (Umeyama 1991).

    Returns ``(s, R, t)`` such that ``s * R @ src.T + t[:, None]`` best
    approximates ``dst.T`` in MSE. ``src`` and ``dst`` must be the same
    shape ``(N, 3)`` with corresponding rows — for ETH3D we feed
    corresponding camera centres (pred vs GT) since the predicted and
    laser-scan point clouds aren't point-corresponding.

    Needs N ≥ 3 non-collinear correspondences to be well-posed; raises
    ValueError otherwise (MASt3R's 2-view case will hit this).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[-1] != 3:
        raise ValueError(f"src/dst must be (N, 3) and matching; got {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"umeyama_similarity needs N >= 3 correspondences; got {n}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Cross-covariance (3, 3).
    H = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_src = float((src_c**2).sum()) / n
    if var_src < EPS:
        # Degenerate source (all points coincide). Fall back to identity
        # rotation and translation-only alignment; scale is undefined.
        return 1.0, np.eye(3, dtype=np.float64), dst_mean - src_mean
    s = float((S * D.diagonal()).sum() / var_src)
    t = dst_mean - s * (R @ src_mean)
    return s, R.astype(np.float64), t.astype(np.float64)


def apply_similarity(
    pts: NDArray[Any],
    s: float,
    R: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply ``X ↦ s * R @ X + t`` to a batch of points.

    Accepts shapes ``(N, 3)`` or ``(..., 3)``; returns float64 of the same
    shape. NaN rows pass through unchanged (the transform of NaN is NaN).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"pts last dim must be 3; got shape {pts.shape}")
    return s * (pts @ R.T) + t


def icp_similarity(
    src: NDArray[Any],
    dst: NDArray[Any],
    *,
    init_s: float | None = None,
    init_R: NDArray[np.float64] | None = None,
    init_t: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    rel_tol: float = 1e-4,
    sample_cap: int | None = 200_000,
    rng_seed: int = 0,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Iterative Closest Point 7-DoF similarity fit of dense ``src`` → dense ``dst``.

    Unlike :func:`umeyama_similarity` — which needs per-row correspondence —
    ICP works on unordered dense point clouds. Each iteration:

      1. For each ``src`` point, find its nearest neighbour in ``dst`` (KDTree).
      2. Reject the top 10% of distances as outliers (trimmed correspondences).
      3. Re-fit (s, R, t) with Umeyama on the remaining inliers.
      4. Apply and repeat until the mean inlier distance stops improving by
         more than ``rel_tol`` relative.

    This is the chamfer-protocol alignment that VGGT / MASt3R / DUSt3R
    papers report on ETH3D and DTU — their published chamfer numbers are
    against ICP-aligned predictions, not camera-centres Umeyama.

    Parameters
    ----------
    src, dst
        ``(N, 3)`` / ``(M, 3)`` float arrays. Caller keeps units consistent
        (no auto-normalization).
    init_s, init_R, init_t
        Optional warm-start similarity (from camera-centres Umeyama) —
        dramatically speeds up convergence when pred and dst are far apart
        in the raw frame. Default: identity.
    max_iter
        Hard cap on iterations (typically converges in 5-15).
    rel_tol
        Relative improvement threshold for the mean inlier distance;
        iteration stops when ``(prev - curr) / prev < rel_tol``.
    sample_cap
        If ``src`` has more points than this, subsample (deterministically
        per ``rng_seed``) before each KDTree query. Bounds wall time on
        the million-point predictions typical of dense MVS. Set to ``None``
        to disable. Default matches the loader's ``max_gt_points`` so the
        final correspondence pool is comparable on both sides.

    Returns
    -------
    s, R, t, info
        Similarity transform plus a dict with convergence stats: final
        mean inlier distance, iteration count, initial vs final.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[-1] != 3 or dst.ndim != 2 or dst.shape[-1] != 3:
        raise ValueError(f"src/dst must be (N, 3); got {src.shape} / {dst.shape}")
    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise ValueError(
            f"icp_similarity needs at least 3 points on each side; got "
            f"{src.shape[0]} / {dst.shape[0]}"
        )

    # Warm start: caller-supplied transform (e.g. from umeyama on camera
    # centres), else identity.
    s = 1.0 if init_s is None else float(init_s)
    R = np.eye(3, dtype=np.float64) if init_R is None else np.asarray(init_R, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64) if init_t is None else np.asarray(init_t, dtype=np.float64)

    # Deterministic subsample of src — large dense predictions have 1M+
    # points and we don't need every one of them to converge ICP.
    if sample_cap is not None and src.shape[0] > sample_cap:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(src.shape[0], size=sample_cap, replace=False)
        src_sub = src[idx]
    else:
        src_sub = src

    from scipy.spatial import cKDTree

    tree = cKDTree(dst)

    prev_mean = float("inf")
    info: dict[str, float] = {}
    final_iter = 0
    for it in range(max_iter):
        final_iter = it + 1
        warped = apply_similarity(src_sub, s, R, t)
        d, nn_idx = tree.query(warped, k=1, workers=-1)
        # Trim the top 10% as outliers — standard ICP robustification.
        keep = d < np.quantile(d, 0.9)
        if keep.sum() < 3:
            break
        src_keep = src_sub[keep]
        dst_keep = dst[nn_idx[keep]]
        s, R, t = umeyama_similarity(src_keep, dst_keep)
        mean_d = float(d[keep].mean())
        if it == 0:
            info["initial_mean_inlier"] = mean_d
        if prev_mean < float("inf") and (prev_mean - mean_d) / max(prev_mean, EPS) < rel_tol:
            break
        prev_mean = mean_d

    info["iterations"] = float(final_iter)
    info["final_mean_inlier"] = prev_mean if prev_mean < float("inf") else float("nan")
    return s, R, t, info
