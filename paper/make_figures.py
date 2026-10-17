"""Generate the paper figures from the canonical website result ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
BLUE = (42, 92, 165)
ORANGE = (218, 124, 48)
INK = (31, 37, 44)
MUTED = (105, 113, 122)
GRID = (220, 224, 229)
PAPER = (251, 250, 247)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "/System/Library/Fonts/Helvetica.ttc"
    return ImageFont.truetype(family, size=size, index=1 if bold else 0)


def _ledger(name: str) -> list[dict[str, Any]]:
    source = (ROOT / "site" / "explore.html").read_text(encoding="utf-8")
    match = re.search(
        rf'<script type="application/json" id="{re.escape(name)}">\n([\s\S]*?)\n</script>',
        source,
    )
    if match is None:
        raise ValueError(f"missing ledger {name}")
    data = json.loads(match.group(1))
    if not isinstance(data, list):
        raise ValueError(f"ledger {name} is not a list")
    return data


def deviation_figure(rows: list[dict[str, Any]]) -> None:
    width, height = 1800, 920
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    title, label, small = _font(54, bold=True), _font(30), _font(24)
    draw.text((100, 55), "Deviation from the published value", fill=INK, font=title)
    draw.text(
        (100, 125),
        "Each point is one verified primary-metric cell; lower is closer.",
        fill=MUTED,
        font=label,
    )
    left, top, right, bottom = 120, 220, 1710, 760
    max_y = 8.0
    for tick in range(0, 9):
        y = bottom - (tick / max_y) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((55, y - 15), f"{tick}%", fill=MUTED, font=small)
    y5 = bottom - 5 / max_y * (bottom - top)
    draw.line((left, y5, right, y5), fill=(176, 71, 71), width=4)
    draw.text((right - 210, y5 - 38), "5% reference", fill=(160, 60, 60), font=small)

    ordered = sorted(rows, key=lambda row: (abs(float(row["d"])), row["model"], row["ds"]))
    for index, row in enumerate(ordered):
        x = left + index / (len(ordered) - 1) * (right - left)
        deviation = abs(float(row["d"]))
        y = bottom - deviation / max_y * (bottom - top)
        color = ORANGE if row.get("task") == "pose" else BLUE
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline=PAPER, width=2)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    draw.text((left, bottom + 28), "cells ordered by absolute deviation", fill=MUTED, font=small)
    draw.ellipse((1180, 105, 1200, 125), fill=BLUE)
    draw.text((1210, 97), "depth (39)", fill=INK, font=small)
    draw.ellipse((1430, 105, 1450, 125), fill=ORANGE)
    draw.text((1460, 97), "pose (8)", fill=INK, font=small)
    draw.text(
        (100, 835),
        "Median 1.5%  |  90th percentile 4.8%  |  46 of 47 cells at or below 5%",
        fill=INK,
        font=label,
    )
    image.save(OUT / "verified_deviation.png", optimize=True)


def coverage_figure(rows: list[dict[str, Any]]) -> None:
    models = sorted({str(row["model"]) for row in rows})
    datasets = sorted({str(row["ds"]) for row in rows})
    width, height = 1900, 1180
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    title, label, small = _font(48, bold=True), _font(25), _font(21)
    draw.text((70, 45), "Verified coverage is broad but not rectangular", fill=INK, font=title)
    draw.text(
        (70, 108),
        "A filled cell means at least one audited primary metric for that method-dataset pair.",
        fill=MUTED,
        font=label,
    )
    left, top = 430, 260
    cell_w, cell_h = 105, 49
    lookup: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        lookup.setdefault((str(row["model"]), str(row["ds"])), set()).add(
            str(row.get("task", "depth"))
        )
    for column, dataset in enumerate(datasets):
        x = left + column * cell_w + 38
        draw.text((x, top - 18), dataset, fill=INK, font=small, anchor="ls", stroke_width=0)
    for row_index, model in enumerate(models):
        y = top + row_index * cell_h
        draw.text((left - 24, y + 24), model, fill=INK, font=small, anchor="rm")
        for column, dataset in enumerate(datasets):
            x = left + column * cell_w
            draw.rectangle((x, y, x + cell_w - 6, y + cell_h - 6), fill=(239, 240, 241))
            tasks = lookup.get((model, dataset), set())
            if tasks:
                color = ORANGE if "pose" in tasks else BLUE
                draw.rounded_rectangle(
                    (x + 7, y + 6, x + cell_w - 13, y + cell_h - 12),
                    radius=8,
                    fill=color,
                )
                count = sum(1 for item in rows if item["model"] == model and item["ds"] == dataset)
                draw.text(
                    (x + (cell_w - 6) / 2, y + 21),
                    str(count),
                    fill="white",
                    font=small,
                    anchor="mm",
                )
    legend_y = top + len(models) * cell_h + 45
    draw.rounded_rectangle((left, legend_y, left + 35, legend_y + 25), radius=5, fill=BLUE)
    draw.text((left + 48, legend_y + 12), "depth", fill=INK, font=small, anchor="lm")
    draw.rounded_rectangle((left + 180, legend_y, left + 215, legend_y + 25), radius=5, fill=ORANGE)
    draw.text((left + 228, legend_y + 12), "pose", fill=INK, font=small, anchor="lm")
    draw.text(
        (left + 390, legend_y + 12),
        "number = verified primary metrics in the pair",
        fill=MUTED,
        font=small,
        anchor="lm",
    )
    image.save(OUT / "coverage_matrix.png", optimize=True)


def pipeline_figure() -> None:
    width, height = 1900, 620
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    title, label, small = _font(48, bold=True), _font(25, bold=True), _font(22)
    draw.text((70, 45), "Protocol-first reproduction method", fill=INK, font=title)
    boxes = [
        ("1", "Paper cell", "table, row, column, metric"),
        ("2", "Pinned protocol", "data slice, crop, alignment, units"),
        ("3", "Public model", "checkpoint, preprocessing, inference"),
        ("4", "Shared scoring", "metric code, cache, report"),
        ("5", "Evidence state", "match, explained gap, blocked"),
    ]
    x0, y0, box_w, box_h, gap = 65, 180, 315, 255, 52
    for index, (number, heading, detail) in enumerate(boxes):
        x = x0 + index * (box_w + gap)
        draw.rounded_rectangle(
            (x, y0, x + box_w, y0 + box_h), radius=18, fill="white", outline=GRID, width=3
        )
        draw.ellipse((x + 22, y0 + 24, x + 72, y0 + 74), fill=BLUE)
        draw.text((x + 47, y0 + 49), number, fill="white", font=label, anchor="mm")
        draw.text((x + 22, y0 + 105), heading, fill=INK, font=label)
        words = detail.split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=small) > box_w - 44:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
        for line_index, line in enumerate(lines):
            draw.text((x + 22, y0 + 155 + line_index * 31), line, fill=MUTED, font=small)
        if index < len(boxes) - 1:
            arrow_x = x + box_w + 10
            draw.line((arrow_x, y0 + box_h / 2, arrow_x + gap - 20, y0 + box_h / 2), fill=ORANGE, width=6)
            draw.polygon(
                [
                    (arrow_x + gap - 20, y0 + box_h / 2 - 13),
                    (arrow_x + gap - 2, y0 + box_h / 2),
                    (arrow_x + gap - 20, y0 + box_h / 2 + 13),
                ],
                fill=ORANGE,
            )
    draw.text(
        (70, 525),
        "Every promoted result keeps the paper target, observed value, tolerance, source confidence, and configuration together.",
        fill=INK,
        font=small,
    )
    image.save(OUT / "method_pipeline.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verified = _ledger("verified-cells")
    if len(verified) != 47:
        raise ValueError(f"paper expects 47 verified cells, found {len(verified)}")
    deviation_figure(verified)
    coverage_figure(verified)
    pipeline_figure()
    print(f"wrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
