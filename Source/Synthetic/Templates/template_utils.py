"""Helpers for deterministic icon-style stroke templates."""

from __future__ import annotations

import math
import random
from typing import Any


Point = tuple[float, float]
Line = tuple[Point, Point]


def build_template_result(
    template: str,
    style_name: str,
    style: dict[str, Any],
    rng: random.Random,
    base_lines: list[Line],
    finishing_lines: list[Line],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    return {
        "template": template,
        "style": style_name,
        "base_strokes": lines_to_strokes(
            lines=expand_lines(base_lines, base_count, rng, jitter=0.018),
            style=style,
            rng=rng,
            role="base",
        ),
        "finishing_strokes": lines_to_strokes(
            lines=expand_lines(finishing_lines, finishing_count, rng, jitter=0.012),
            style=style,
            rng=rng,
            role="finishing",
        ),
    }


def expand_lines(lines: list[Line], target_count: int, rng: random.Random, jitter: float) -> list[Line]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not lines:
        raise ValueError("template line list must not be empty")

    expanded: list[Line] = []
    index = 0
    while len(expanded) < target_count:
        line = lines[index % len(lines)]
        if index < len(lines):
            expanded.append(line)
        else:
            expanded.append(jitter_line(line, rng, jitter))
        index += 1
    return expanded


def jitter_line(line: Line, rng: random.Random, amount: float) -> Line:
    return (
        jitter_point(line[0], rng, amount),
        jitter_point(line[1], rng, amount),
    )


def jitter_point(point: Point, rng: random.Random, amount: float) -> Point:
    return (
        clamp(point[0] + rng.uniform(-amount, amount), 0.03, 0.97),
        clamp(point[1] + rng.uniform(-amount, amount), 0.03, 0.97),
    )


def lines_to_strokes(lines: list[Line], style: dict[str, Any], rng: random.Random, role: str) -> list[dict[str, Any]]:
    return [line_to_stroke(line=line, style=style, rng=rng, role=role) for line in lines]


def line_to_stroke(line: Line, style: dict[str, Any], rng: random.Random, role: str) -> dict[str, Any]:
    (x1, y1), (x2, y2) = line
    dx = x2 - x1
    dy = y2 - y1
    length = max(math.sqrt(dx * dx + dy * dy), 0.002)
    angle = (math.atan2(dy, dx) / (2.0 * math.pi)) % 1.0
    width_range = style["width"]
    opacity_range = style["opacity"]
    width_scale = 1.0 if role == "base" else 0.72
    color = jitter_color(rng.choice(style["palette"]), rng)

    return {
        "x": rounded((x1 + x2) / 2.0),
        "y": rounded((y1 + y2) / 2.0),
        "angle": rounded(angle),
        "length": rounded(clamp(length, 0.001, 1.0)),
        "width": rounded(clamp(rng.uniform(*width_range) * width_scale, 0.001, 1.0)),
        "color": [rounded(channel) for channel in color],
        "opacity": rounded(rng.uniform(*opacity_range)),
        "brush": style["brush"],
    }


def jitter_color(color: list[float], rng: random.Random) -> list[float]:
    return [clamp(channel + rng.uniform(-0.025, 0.025)) for channel in color]


def box_lines(left: float, top: float, right: float, bottom: float) -> list[Line]:
    return [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ]


def box_fill_lines(left: float, top: float, right: float, bottom: float, count: int) -> list[Line]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [((left, (top + bottom) / 2.0), (right, (top + bottom) / 2.0))]
    lines: list[Line] = []
    for index in range(count):
        t = index / (count - 1)
        y = top + (bottom - top) * t
        lines.append(((left, y), (right, y)))
    return lines


def triangle_fill_lines(apex: Point, left_base: Point, right_base: Point, count: int) -> list[Line]:
    if count <= 0:
        raise ValueError("count must be positive")
    lines: list[Line] = []
    for index in range(count):
        t = (index + 1) / (count + 1)
        left = interpolate_point(apex, left_base, t)
        right = interpolate_point(apex, right_base, t)
        lines.append((left, right))
    return lines


def interpolate_point(start: Point, end: Point, t: float) -> Point:
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )


def regular_polygon_lines(cx: float, cy: float, radius: float, sides: int, rotation: float = 0.0) -> list[Line]:
    points = [
        (
            cx + math.cos(rotation + 2.0 * math.pi * index / sides) * radius,
            cy + math.sin(rotation + 2.0 * math.pi * index / sides) * radius,
        )
        for index in range(sides)
    ]
    return [(points[index], points[(index + 1) % sides]) for index in range(sides)]


def radial_lines(cx: float, cy: float, inner: float, outer: float, count: int, rotation: float = 0.0) -> list[Line]:
    lines: list[Line] = []
    for index in range(count):
        angle = rotation + 2.0 * math.pi * index / count
        lines.append(
            (
                (cx + math.cos(angle) * inner, cy + math.sin(angle) * inner),
                (cx + math.cos(angle) * outer, cy + math.sin(angle) * outer),
            )
        )
    return lines


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def rounded(value: float) -> float:
    return round(value, 5)
