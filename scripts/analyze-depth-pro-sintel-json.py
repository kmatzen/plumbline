#!/usr/bin/env python3
"""Per-scene breakdown from a plumbline reproduce JSON (Depth Pro Sintel)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    p.add_argument("--top", type=int, default=10, help="worst/best scenes to print")
    args = p.parse_args()

    data = json.loads(args.json_path.read_text())
    per = data["per_sample"]
    by_scene: dict[str, list[float]] = defaultdict(list)
    frame_order: list[float] = []

    for row in per:
        if row.get("skipped"):
            continue
        sid = row["sample_id"]
        scene = sid.split("/")[0]
        d1 = float(row["metrics"]["delta_1"])
        by_scene[scene].append(d1)
        frame_order.append(d1)

    def _mean(xs: list[float]) -> float:
        fin = [x for x in xs if math.isfinite(x)]
        return sum(fin) / len(fin) if fin else float("nan")

    scene_mean = {s: _mean(v) for s, v in by_scene.items()}
    sorted_scenes = sorted(scene_mean.items(), key=lambda x: x[1])

    agg = data["aggregate_metrics"]["delta_1"]
    fin_frames = [x for x in frame_order if math.isfinite(x)]
    equal_frame_mean = _mean(frame_order)
    scene_fin = [m for m in scene_mean.values() if math.isfinite(m)]
    equal_scene_mean = sum(scene_fin) / len(scene_fin) if scene_fin else float("nan")
    n_nan = len(frame_order) - len(fin_frames)

    n = len(frame_order)
    head = [x for x in frame_order[:80] if math.isfinite(x)]
    tail = [x for x in frame_order[80:] if math.isfinite(x)]
    print(f"JSON: {args.json_path}")
    print(f"aggregate delta_1: {agg:.4f}  (n={n}, nan_frames={n_nan})")
    print(f"equal-frame mean:   {equal_frame_mean:.4f}")
    print(f"equal-scene mean:   {equal_scene_mean:.4f}  ({len(scene_mean)} scenes)")
    if head:
        print(f"first 80 frames:    {_mean(head):.4f}  (finite={len(head)})")
    if tail:
        print(f"frames 81-{n}:        {_mean(tail):.4f}  (finite={len(tail)})")
    worst = sorted((s for s, m in scene_mean.items() if m < 0.05), key=lambda s: scene_mean[s])
    if worst:
        ex = [x for s in worst for x in by_scene[s] if math.isfinite(x)]
        rest = [x for s, vs in by_scene.items() if s not in worst for x in vs if math.isfinite(x)]
        print(f"excl {len(worst)} scenes <0.05:  frame_mean={_mean(rest):.4f}  (n={len(rest)})")
    print()
    print(f"Worst {args.top} scenes (low delta_1):")
    for s, m in sorted_scenes[: args.top]:
        print(f"  {s:20s}  delta_1={m:.4f}  n={len(by_scene[s])}")
    print(f"\nBest {args.top} scenes:")
    for s, m in sorted_scenes[-args.top :][::-1]:
        print(f"  {s:20s}  delta_1={m:.4f}  n={len(by_scene[s])}")


if __name__ == "__main__":
    main()
