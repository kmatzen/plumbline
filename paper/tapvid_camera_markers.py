"""Camera marker geometry and separately applied point-reference labels."""

import numpy as np

from paper.build_anchor_preview import fit_point
from paper.build_tapvid_judge_corpus import target_label

LAGS = (15, 30, 60)


def project(point, pose, calibration, size):
    camera = np.linalg.solve(np.asarray(pose, float), np.r_[point, 1.])[:3]
    if camera[2] <= 0:
        return None
    uv = calibration @ camera
    return uv[:2]/uv[2] * size - .5


def fit_marker(track, poses, calibration, size):
    """Consume source query and nine early RGB estimates, never later labels."""
    first = track.get("query_frame")
    if first is None:
        return dict(status="no_visible_query")
    indices = list(range(first, first+10))
    if indices[-1] >= len(poses):
        return dict(status="prefix_beyond_video")
    observations = {r["frame"]:r["xy"] for r in track["records"] if first < r["frame"] <= first+9}
    observations[first] = track["query_xy"]
    if any(observations.get(i) is None for i in indices):
        return dict(status="missing_prefix_observations")
    xy = np.array([observations[i] for i in indices], dtype=float)
    point = fit_point(xy, poses[indices], calibration[indices], np.asarray(size))
    if point is None or not np.isfinite(point).all():
        return dict(status="degenerate_prefix_fit")
    projected = [project(point, poses[i], calibration[i], size) for i in indices]
    if any(p is None for p in projected):
        return dict(status="nonpositive_prefix_depth", point_world=point.tolist())
    return dict(status="fitted", point_world=point.tolist(), source_frame=first,
                source_xy=track["query_xy"], fit_frames=indices,
                prefix_max_pixel_error=float(np.linalg.norm(np.array(projected)-xy, axis=1).max()))


def proposals(track, poses, calibration, size):
    """Freeze all candidate positions before independent annotation scoring."""
    fit = fit_marker(track, poses, calibration, size)
    first = track.get("query_frame")
    rows = []
    for lag in LAGS:
        frame = None if first is None else first+lag
        row = dict(lag=lag, frame=frame, candidate_xy=None)
        if first is None:
            row["status"] = "no_visible_query"
        elif frame >= len(poses):
            row["status"] = "query_beyond_video"
        elif fit["status"] != "fitted":
            row["status"] = fit["status"]
        else:
            xy = project(np.array(fit["point_world"]), poses[frame], calibration[frame], size)
            if xy is None:
                row["status"] = "nonpositive_query_depth"
            elif not np.isfinite(xy).all():
                row["status"] = "nonfinite_query_projection"
            else:
                row["candidate_xy"] = xy.tolist()
                rounded = np.rint(xy)
                row["status"] = ("renderable" if np.all((rounded >= 0) & (rounded < size))
                                 else "query_outside_image")
        rows.append(row)
    return fit, rows


def label_proposals(rows, normalized_points, occluded, size):
    """Annotations may assign scoring labels but cannot change candidates."""
    labeled = []
    for row in rows:
        result = dict(row, label=None)
        if row["status"] == "renderable":
            frame = row["frame"]
            xy = np.asarray(normalized_points[frame])*size
            label = target_label(row["candidate_xy"], xy, bool(occluded[frame]))
            result.update(label=label, scoring_status="labeled" if label else "borderline_unlabeled",
                          reference_visible=not bool(occluded[frame]),
                          reference_distance_native_pixels=(None if occluded[frame] else
                              float(np.linalg.norm(np.asarray(row["candidate_xy"])-xy))))
        else:
            result["scoring_status"] = "unavailable"
        labeled.append(result)
    return labeled
