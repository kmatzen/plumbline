"""Reproductions: pinned paper-number configs.

A reproduction is a YAML file under ``reproductions/<name>.yaml`` declaring:

- Which model + version + preprocessing knobs to use.
- Which dataset + split + sample list.
- Which tasks, scale alignment, view count, resolution.
- The published reference: metric, value, and tolerance.
- Optionally ``min_samples``: a floor on the number of samples the exact
  eval set should produce. If the run evaluates fewer, the reproduction is
  forced to ``paper_match=no`` with a COUNT SHORTFALL note — guards against
  a sparse / mis-pointed data root landing inside tolerance by luck (D28).

Running a reproduction executes :func:`~plumbline.runner.evaluate` with those
settings and compares the primary metric against the published value, within
tolerance.
"""

from __future__ import annotations

import importlib.resources as resources
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plumbline.cache import PredictionCache
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.protocols import apply_protocol
from plumbline.report import Report
from plumbline.runner import evaluate

__all__ = [
    "REPRODUCTIONS_DIR",
    "ReproductionResult",
    "expected_metric_keys",
    "load_reproduction_config",
    "run_reproduction",
]

REPRODUCTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "reproductions"

log = logging.getLogger(__name__)


def expected_metric_keys(cfg: dict[str, Any]) -> set[str]:
    """Static over-approximation of the metric keys a config's run can emit.

    Mirrors the key construction in :func:`plumbline.runner.evaluate` /
    ``_compute_metrics`` so a reproduction's ``paper_reference.primary_metric``
    can be validated at load time — *before* burning a GPU run that would
    otherwise silently report ``NaN`` via ``aggregate_metrics.get(metric, nan)``
    (the pi3-dtu 'chamfer' vs 'overall' class of bug). ``cfg`` must be
    protocol-merged (post :func:`apply_protocol`).

    Conservative by design: returns the *union* of keys reachable for the
    declared tasks/aggregation, so it never rejects a valid metric. Keep in
    sync with the runner; ``tests/test_reproduction_metric_keys.py`` enforces
    that every shipped reproduction's primary_metric is in this set.
    """
    tasks = set(cfg.get("tasks", []))
    # Scene aggregation replaces the per-sample metric path entirely
    # (runner: the scene branch ``continue``s before _compute_metrics).
    if cfg.get("aggregation", "sample") == "scene":
        return {
            "overall",
            "accuracy",
            "completeness",
            "overall_median",
            "accuracy_median",
            "completeness_median",
        }
    keys: set[str] = set()
    if {"mono_depth", "mvs_depth"} & tasks:
        keys |= {"abs_rel", "rmse", "log10", "silog"}
        deltas = cfg.get("delta_thresholds") or (1.25, 1.25**2, 1.25**3)
        keys |= {f"delta_{i}" for i in range(1, len(deltas) + 1)}
    if "pose" in tasks:
        auc_thr = cfg.get("pose_auc_thresholds") or (5.0, 10.0, 30.0)
        acc_thr = cfg.get("pose_acc_thresholds") or (15.0,)
        keys |= {"rotation_error_deg_mean", "translation_cos_err_deg_mean"}
        keys |= {"pairwise_rot_err_deg_mean", "pairwise_trans_cos_err_deg_mean"}
        for t in auc_thr:
            keys.add(f"pose_auc@{float(t):g}")
            keys.add(f"pairwise_pose_auc@{float(t):g}")
        for t in acc_thr:
            keys.add(f"pairwise_RRA@{float(t):g}")
            keys.add(f"pairwise_RTA@{float(t):g}")
        if cfg.get("pose_trajectory_metrics"):
            keys |= {
                "trajectory_ate_rmse",
                "trajectory_rpe_trans_rmse",
                "trajectory_rpe_rot_deg_rmse",
            }
    if {"mvs_depth", "point_cloud"} & tasks:
        keys |= {"chamfer", "precision", "recall", "f_score"}
    return keys


@dataclass
class ReproductionResult:
    name: str
    report: Report
    primary_metric: str
    observed: float
    published: float
    tolerance_relative: float
    paper_match: bool | None
    notes: str = ""
    n_evaluated: int = 0
    min_samples: int | None = None

    @property
    def count_shortfall(self) -> bool:
        """True when a declared ``min_samples`` floor was not met.

        Guards the D28 footgun: a sparse / mis-pointed data root that
        silently evaluates far fewer frames than the protocol declares can
        still land a metric inside tolerance by luck (the KITTI 82-frame
        run that coincidentally matched 0.1086). When the floor is missed,
        the metric was computed on the wrong sample set and must not count
        as a paper match.
        """
        return self.min_samples is not None and self.n_evaluated < self.min_samples

    def to_markdown(self) -> str:
        md = self.report.to_markdown()
        md += "\n## Reproduction check\n\n"
        md += f"- Primary metric: `{self.primary_metric}`\n"
        md += f"- Observed: {self.observed:.4f}\n"
        md += f"- Published: {self.published:.4f}\n"
        md += f"- Tolerance (relative): {self.tolerance_relative:.2%}\n"
        if self.min_samples is not None:
            ok = "" if not self.count_shortfall else "  ⚠️ BELOW MINIMUM"
            md += f"- Samples evaluated: {self.n_evaluated} (min {self.min_samples}){ok}\n"
        if self.paper_match is not None:
            md += f"- Match: **{'yes' if self.paper_match else 'no'}**\n"
        if self.notes:
            md += f"\n> {self.notes}\n"
        return md


def load_reproduction_config(name: str) -> dict[str, Any]:
    """Load a reproduction YAML by short name (e.g. ``vggt-paper-scannet-depth``)."""
    path = _find_config(name)
    if path is None:
        raise FileNotFoundError(
            f"No reproduction config for '{name}'. Looked under {REPRODUCTIONS_DIR} "
            f"and the installed package."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_reproduction(name: str, *, output: Path | None = None) -> ReproductionResult:
    # Import built-in adapters so programmatic callers don't have to
    # remember to eager-register before looking things up.
    from plumbline._discover import register_builtin_adapters

    register_builtin_adapters()

    cfg = load_reproduction_config(name)
    # Resolve `protocol:` if set — merges the protocol's fixed fields
    # into the reproduction config and raises on conflict. Identity op
    # for YAMLs that don't declare a protocol.
    cfg = apply_protocol(cfg)

    # Fail fast (before the GPU run) if the declared primary_metric can't be
    # emitted by this config's path — otherwise the run silently reports NaN
    # via ``aggregate_metrics.get(primary_metric, nan)``. See expected_metric_keys.
    _pm = cfg.get("paper_reference", {}).get("primary_metric")
    if _pm is not None and _pm not in expected_metric_keys(cfg):
        raise ValueError(
            f"reproduction {name!r}: paper_reference.primary_metric {_pm!r} is not "
            f"among the metrics this config can emit ({sorted(expected_metric_keys(cfg))}). "
            f"Fix the yaml — a scene-aggregation run emits 'overall', not 'chamfer'."
        )

    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"model '{model_name}' not registered")
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"dataset '{dataset_name}' not registered")

    model_cls = MODEL_REGISTRY[model_name]
    dataset_cls = DATASET_REGISTRY[dataset_name]

    model_kwargs: dict[str, Any] = {"device": cfg.get("device", "cuda:0")}
    model_kwargs.update(cfg["model"].get("kwargs", {}))
    model = model_cls(**model_kwargs)

    dataset_kwargs: dict[str, Any] = dict(cfg["dataset"].get("kwargs", {}))
    if "split" in cfg["dataset"]:
        dataset_kwargs.setdefault("split", cfg["dataset"]["split"])
    dataset = dataset_cls(**dataset_kwargs)

    # Sample-list pinning takes precedence over numeric subset — reproductions
    # need exact samples, not a stride over an evolving manifest.
    sample_ids_file = cfg.get("sample_ids_file")
    if sample_ids_file:
        ids_path = Path(sample_ids_file)
        if not ids_path.is_absolute():
            ids_path = REPRODUCTIONS_DIR / ids_path
        sample_ids = _read_sample_ids(ids_path)
        dataset = dataset.subset_by_ids(sample_ids)
    else:
        subset_n = cfg.get("subset")
        if subset_n:
            dataset = dataset.subset(int(subset_n))

    depth_clip_cfg = cfg.get("depth_clip")
    depth_clip = (float(depth_clip_cfg[0]), float(depth_clip_cfg[1])) if depth_clip_cfg else None

    pose_auc_thresholds = tuple(float(t) for t in cfg.get("pose_auc_thresholds", (5.0, 10.0, 30.0)))
    pose_acc_thresholds = tuple(float(t) for t in cfg.get("pose_acc_thresholds", (15.0,)))
    pose_auc_mode = str(cfg.get("pose_auc_mode", "analytic"))
    pose_translation_antipodal = bool(cfg.get("pose_translation_antipodal", False))
    pose_trajectory_metrics = bool(cfg.get("pose_trajectory_metrics", False))

    report = evaluate(
        model=model,
        dataset=dataset,
        tasks=list(cfg["tasks"]),
        scale_alignment=cfg.get("scale_alignment", "median"),
        max_views=int(cfg.get("max_views", 8)),
        device=cfg.get("device", "cuda:0"),
        cache=PredictionCache(cfg.get("cache_dir")) if cfg.get("cache_dir") else None,
        pose_auc_thresholds=pose_auc_thresholds,
        pose_acc_thresholds=pose_acc_thresholds,
        pose_auc_mode=pose_auc_mode,
        pose_translation_antipodal=pose_translation_antipodal,
        pose_trajectory_metrics=pose_trajectory_metrics,
        depth_clip=depth_clip,
        pointcloud_alignment=cfg.get("pointcloud_alignment", "none"),
        chamfer_outlier_distance=cfg.get("chamfer_outlier_distance"),
        mask_boundaries=bool(cfg.get("mask_boundaries", False)),
        boundary_thickness=int(cfg.get("boundary_thickness", 1)),
        boundary_tol=float(cfg.get("boundary_tol", 0.1)),
        aggregation=cfg.get("aggregation", "sample"),
        scene_voxel_size=float(cfg.get("scene_voxel_size", 0.01)),
        per_view_masked=bool(cfg.get("per_view_masked", False)),
        per_view_crop=int(cfg.get("per_view_crop", 224)),
        geometric_consistency=bool(cfg.get("geometric_consistency", False)),
        geo_pixel_thres=float(cfg.get("geo_pixel_thres", 1.0)),
        geo_depth_thres=float(cfg.get("geo_depth_thres", 0.01)),
        geo_mask_thres=int(cfg.get("geo_mask_thres", 3)),
        geo_num_src_views=int(cfg.get("geo_num_src_views", 4)),
    )

    paper = cfg.get("paper_reference", {})
    primary_metric = paper.get("primary_metric") or next(iter(report.aggregate_metrics))
    if primary_metric not in report.aggregate_metrics:
        # A primary_metric the protocol never emits silently produced a NaN
        # ``observed`` (pi3-dtu asked for 'chamfer' while the MVS path emits
        # 'overall'/'accuracy'/'completeness'). Surface it loudly so a yaml
        # typo is a visible error, not a phantom NaN reproduction.
        log.warning(
            "reproduction %r: primary_metric %r is not among the computed "
            "metrics %s; observed will be NaN. Fix paper_reference.primary_metric "
            "in the reproduction yaml to match a metric the protocol emits.",
            name,
            primary_metric,
            sorted(report.aggregate_metrics),
        )
    observed = float(report.aggregate_metrics.get(primary_metric, float("nan")))
    # value / tolerance_relative may be null (intentionally — e.g. D16: indoor-only run,
    # paper cites combined-val). Treat null / missing as NaN / default so paper_match
    # becomes None (informational) rather than crashing.
    published_raw = paper.get("value")
    published = float(published_raw) if published_raw is not None else float("nan")
    tolerance_raw = paper.get("tolerance_relative")
    tolerance = float(tolerance_raw) if tolerance_raw is not None else 0.05

    match: bool | None
    if published == published and observed == observed:  # both non-NaN
        match = abs(observed - published) / max(abs(published), 1e-8) <= tolerance
    else:
        match = None

    # Optional sample-count floor (D28 footgun guard). A reproduction may
    # declare ``min_samples`` — the lower bound on frames the protocol's
    # exact eval set should produce. If the run evaluated fewer (sparse or
    # mis-pointed data root, an upstream loader silently under-counting),
    # the metric is off the declared set and cannot count as a match, even
    # if it lands inside tolerance by luck.
    min_samples_raw = cfg.get("min_samples")
    min_samples = int(min_samples_raw) if min_samples_raw is not None else None
    notes = cfg.get("notes", "")
    if min_samples is not None and report.n_evaluated < min_samples:
        match = False
        shortfall = (
            f"COUNT SHORTFALL: evaluated {report.n_evaluated} samples but the protocol "
            f"declares min_samples={min_samples}. The metric was computed on the wrong "
            f"set — check the data root / sample selection. Forced paper_match=no."
        )
        log.warning("reproduction %r: %s", name, shortfall)
        notes = f"{shortfall}\n\n{notes}" if notes else shortfall

    result = ReproductionResult(
        name=name,
        report=report,
        primary_metric=primary_metric,
        observed=observed,
        published=published,
        tolerance_relative=tolerance,
        paper_match=match,
        notes=notes,
        n_evaluated=report.n_evaluated,
        min_samples=min_samples,
    )

    if output:
        report.save_json(output)
    return result


def _read_sample_ids(path: Path) -> list[str]:
    """Read a newline-delimited sample-id list.

    Lines starting with ``#`` and empty lines are ignored. Used by
    reproductions to pin the exact sample set across dataset re-scans.
    """
    if not path.exists():
        raise FileNotFoundError(f"sample_ids_file not found: {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    if not ids:
        raise ValueError(f"sample_ids_file is empty: {path}")
    return ids


def _find_config(name: str) -> Path | None:
    candidates = [
        REPRODUCTIONS_DIR / f"{name}.yaml",
        REPRODUCTIONS_DIR / f"{name.replace('-', '_')}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    try:
        pkg = resources.files("plumbline").parent / "reproductions"  # type: ignore[attr-defined]
        for suffix in (f"{name}.yaml", f"{name.replace('-', '_')}.yaml"):
            candidate = pkg / suffix
            if candidate.is_file():
                return Path(str(candidate))
    except Exception:
        pass
    return None
