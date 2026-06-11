from __future__ import annotations

import random
from typing import Any

from Source.Synthetic.Templates.template_utils import build_template_result, regular_polygon_lines


def generate_face_icon(
    rng: random.Random,
    style_name: str,
    style: dict[str, Any],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    base_lines = [
        *regular_polygon_lines(0.50, 0.50, 0.30, 12, rotation=0.1),
        ((0.36, 0.36), (0.43, 0.34)),
        ((0.57, 0.34), (0.64, 0.36)),
        ((0.38, 0.66), (0.50, 0.72)),
        ((0.50, 0.72), (0.62, 0.66)),
    ]
    finishing_lines = [
        ((0.38, 0.43), (0.44, 0.43)),
        ((0.56, 0.43), (0.62, 0.43)),
        ((0.41, 0.40), (0.41, 0.46)),
        ((0.59, 0.40), (0.59, 0.46)),
        ((0.50, 0.46), (0.47, 0.57)),
        ((0.47, 0.57), (0.53, 0.57)),
        ((0.41, 0.62), (0.50, 0.67)),
        ((0.50, 0.67), (0.59, 0.62)),
        ((0.31, 0.50), (0.24, 0.54)),
        ((0.69, 0.50), (0.76, 0.54)),
    ]
    return build_template_result("face_icon", style_name, style, rng, base_lines, finishing_lines, base_count, finishing_count)

