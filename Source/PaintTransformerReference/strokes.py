"""Conversion between Paint Transformer stroke params and BrushWright strokes."""

from __future__ import annotations

from typing import Any


PAINT_TRANSFORMER_BRUSH = "paint_transformer_rect"
MIN_STROKE_SIZE = 0.001


def paint_transformer_param_to_stroke(
    param: list[float],
    patch_x: int,
    patch_y: int,
    patch_count: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict[str, Any]:
    if len(param) != 8:
        raise ValueError("Paint Transformer stroke params must contain 8 values")
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")

    x_center, y_center, width, height, theta, red, green, blue = param
    global_x = _clamp01((patch_x + offset_x + x_center) / patch_count)
    global_y = _clamp01((patch_y + offset_y + y_center) / patch_count)
    normalized_width = max(MIN_STROKE_SIZE, abs(width) / patch_count)
    normalized_height = max(MIN_STROKE_SIZE, abs(height) / patch_count)
    angle = (theta / 2.0) % 1.0
    if normalized_height > normalized_width:
        angle = (angle + 0.25) % 1.0

    return {
        "x": global_x,
        "y": global_y,
        "angle": _clamp01(angle),
        "length": _clamp01(max(normalized_width, normalized_height)),
        "width": _clamp01(min(normalized_width, normalized_height)),
        "color": [_clamp01(red), _clamp01(green), _clamp01(blue)],
        "opacity": 1.0,
        "brush": PAINT_TRANSFORMER_BRUSH,
    }


def collect_brushwright_strokes(param, decision, patch_count: int, offset_x: float = 0.0, offset_y: float = 0.0):
    strokes: list[dict[str, Any]] = []
    param_values = param.detach().cpu().tolist()
    decision_values = decision.detach().cpu().bool().tolist()
    _, patch_h, patch_w, stroke_count, _ = param.shape
    for patch_y in range(patch_h):
        for patch_x in range(patch_w):
            for stroke_index in range(stroke_count):
                if decision_values[0][patch_y][patch_x][stroke_index]:
                    strokes.append(
                        paint_transformer_param_to_stroke(
                            param=param_values[0][patch_y][patch_x][stroke_index],
                            patch_x=patch_x,
                            patch_y=patch_y,
                            patch_count=patch_count,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        )
                    )
    return strokes


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
