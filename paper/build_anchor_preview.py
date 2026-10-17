"""Development overlay: fit a static point early, project without later refitting.

Image tracks are fallible observations, not ground truth. Background labels do
not establish a static scene. This preview is not a DNF or accuracy benchmark.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest, load_verified
from PIL import Image

from plumbline.image_evidence import rays_from_normalized_intrinsics


def fit_point(pixels, poses, intrinsics, size):
    """Fit world point to rays from supplied training observations only."""
    directions = []
    for xy, pose, k in zip(pixels, poses, intrinsics, strict=True):
        ray = np.r_[rays_from_normalized_intrinsics(xy[None], k, size)[0], 1]
        direction = pose[:3, :3] @ ray
        directions.append(direction / np.linalg.norm(direction))
    directions = np.asarray(directions)
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    matrix = projectors.sum(axis=0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
        return None
    return np.linalg.solve(matrix, np.einsum("nij,nj->i", projectors, poses[:, :3, 3]))


def project(point, pose, k, size):
    camera = pose[:3, :3].T @ (point - pose[:3, 3])
    if camera[2] <= 0:
        return None
    uv = k @ camera
    return uv[:2] / uv[2] * size - 0.5


def build(run, root, mask_path, output, roi=None):
    if output.exists():
        raise ValueError("preserve existing previews")
    manifest, poses, paths = load_verified(run, root)
    with np.load(run / "prediction.npz", allow_pickle=False) as archive:
        intrinsics = archive["intrinsics_normalized"]
    frames = [cv2.imread(str(path)) for path in paths]
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    height, width = gray[0].shape
    size = np.array([width, height])
    mask_record = json.loads((mask_path.parent / "masks.json").read_text())
    first_mask = mask_record["frames"][0]
    if (
        mask_record["sequence"] != manifest["sequence"]
        or first_mask["frame_index"] != manifest["frames"][0]["frame_index"]
        or first_mask["path"] != mask_path.name
        or first_mask["sha256"] != digest(mask_path)
    ):
        raise ValueError("mask provenance mismatch")
    with Image.open(mask_path) as annotation:
        mask = np.array(annotation)
    if mask.shape != gray[0].shape:
        raise ValueError("matching first-frame annotation required")
    eligible = cv2.erode((mask == 0).astype(np.uint8) * 255, np.ones((21, 21), np.uint8))
    if roi is not None:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI must be inside the first image")
        region = np.zeros_like(eligible)
        region[y0:y1, x0:x1] = 255
        eligible &= region
    # Strongest first-frame background corner, chosen without consulting poses,
    # later tracking quality, or later projection errors.
    first = cv2.goodFeaturesToTrack(gray[0], 1, 0.01, 30, mask=eligible)
    if first is None:
        raise ValueError("no eligible first-frame feature")
    track = [first.reshape(2).astype(float)]
    alive = True
    fb_errors = [0.0]
    for index in range(1, len(frames)):
        if not alive:
            track.append(None)
            fb_errors.append(None)
            continue
        previous = np.asarray(track[-1], np.float32).reshape(1, 1, 2)
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[index - 1], gray[index], previous, None, winSize=(21, 21), maxLevel=3
        )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[index], gray[index - 1], current, None, winSize=(21, 21), maxLevel=3
        )
        fb = float(np.linalg.norm(backward - previous))
        xy = current.reshape(2)
        alive = bool(
            status.item()
            and backward_status.item()
            and fb <= 1.0
            and np.isfinite(xy).all()
            and (xy >= 0).all()
            and (xy < size).all()
        )
        track.append(xy.astype(float) if alive else None)
        fb_errors.append(fb)
    fit_count = 11
    point = (
        fit_point(np.array(track[:fit_count]), poses[:fit_count], intrinsics[:fit_count], size)
        if all(x is not None for x in track[:fit_count])
        else None
    )
    if point is not None and any(
        project(point, poses[i], intrinsics[i], size) is None for i in range(fit_count)
    ):
        point = None
    output.mkdir(parents=True)
    rendered = output / "frames"
    rendered.mkdir()
    rows = []
    for i, frame in enumerate(frames):
        predicted = project(point, poses[i], intrinsics[i], size) if point is not None else None
        observed = track[i]
        visible = (
            predicted is not None
            and np.isfinite(predicted).all()
            and (predicted >= 0).all()
            and (predicted < size).all()
        )
        canvas = cv2.copyMakeBorder(frame, 96, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        label = f"Frame {i}: {'FIT (not a test)' if i < fit_count else 'TEST (point fixed)'}"
        cv2.putText(
            canvas, label, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "DOT: image feature   RING: camera prediction",
            (18, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        note = "Does the ring stay on the same physical detail?"
        if point is None:
            note = "No usable early-frame fit; no camera judgment."
        elif observed is None:
            note = "Image track lost; comparison unavailable."
        elif not visible:
            note = "Predicted marker is off-screen or behind the camera."
        cv2.putText(
            canvas, note, (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        if observed is not None:
            dot = tuple(np.rint(observed + np.array([0, 96])).astype(int))
            cv2.circle(canvas, dot, 5, (255, 255, 0), -1, cv2.LINE_AA)
        if visible:
            ring = tuple(np.rint(predicted + np.array([0, 96])).astype(int))
            cv2.circle(canvas, ring, 12, (0, 140, 255), 2, cv2.LINE_AA)
            if observed is not None:
                cv2.line(canvas, dot, ring, (255, 255, 255), 1, cv2.LINE_AA)
        if observed is not None:
            cx, cy = np.rint(observed).astype(int)
            x0, y0 = max(0, cx - 35), max(0, cy - 35)
            x1, y1 = min(width, cx + 35), min(height, cy + 35)
            crop = canvas[y0 + 96 : y1 + 96, x0:x1].copy()
            zoom = cv2.resize(crop, (210, 210), interpolation=cv2.INTER_NEAREST)
            canvas[-220:-10, -220:-10] = zoom
            cv2.rectangle(
                canvas,
                (width - 220, height + 96 - 220),
                (width - 10, height + 96 - 10),
                (255, 255, 255),
                1,
            )
            cv2.putText(
                canvas,
                "ZOOM follows dot",
                (width - 220, height + 96 - 228),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(rendered / f"{i:06d}.png"), canvas)
        rows.append(
            {
                "frame": i,
                "fit_frame": i < fit_count,
                "observed_xy": observed.tolist() if observed is not None else None,
                "predicted_xy": predicted.tolist() if predicted is not None else None,
                "forward_backward_pixels": fb_errors[i],
                "distance_pixels": float(np.linalg.norm(predicted - observed))
                if predicted is not None and observed is not None
                else None,
            }
        )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(rendered / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output / "preview.mp4"),
        ],
        check=True,
    )
    record = {
        "source_manifest_sha256": digest(run / "manifest.json"),
        "mask_sha256": digest(mask_path),
        "script_sha256": digest(Path(__file__)),
        "video_sha256": digest(output / "preview.mp4"),
        "fit_frames": list(range(fit_count)),
        "point_world": point.tolist() if point is not None else None,
        "selection": "strongest first-frame corner in eroded background mask and optional manual ROI; no pose-dependent selection",
        "manual_first_frame_roi_xyxy": roi,
        "scope": "development evidence preview; tracks and background labels are not verified static ground truth",
        "display_fps": 10,
        "capture_timing": None,
        "rows": rows,
    }
    (output / "preview.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "root", "mask", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--roi", type=int, nargs=4)
    args = parser.parse_args()
    build(args.run, args.root, args.mask, args.output, args.roi)
