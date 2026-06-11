from __future__ import annotations

import random
from typing import Any

from Source.Synthetic.Templates.template_utils import box_fill_lines, box_lines, build_template_result, triangle_fill_lines


def generate_tree_icon(
    rng: random.Random,
    style_name: str,
    style: dict[str, Any],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    base_lines = [
        *triangle_fill_lines((0.50, 0.18), (0.26, 0.52), (0.74, 0.52), 12),
        *triangle_fill_lines((0.50, 0.30), (0.32, 0.62), (0.68, 0.62), 10),
        *box_fill_lines(0.45, 0.55, 0.55, 0.80, 8),
        ((0.43, 0.58), (0.57, 0.58)),
        ((0.42, 0.64), (0.58, 0.64)),
        ((0.44, 0.72), (0.56, 0.72)),
    ]
    finishing_lines = [
        ((0.50, 0.18), (0.26, 0.52)),
        ((0.50, 0.18), (0.74, 0.52)),
        ((0.30, 0.52), (0.70, 0.52)),
        ((0.50, 0.30), (0.32, 0.62)),
        ((0.50, 0.30), (0.68, 0.62)),
        ((0.36, 0.62), (0.64, 0.62)),
        *box_lines(0.45, 0.55, 0.55, 0.80),
        ((0.50, 0.22), (0.50, 0.52)),
        ((0.50, 0.38), (0.40, 0.48)),
        ((0.50, 0.42), (0.61, 0.51)),
        ((0.50, 0.54), (0.39, 0.60)),
        ((0.50, 0.56), (0.62, 0.61)),
        ((0.44, 0.70), (0.56, 0.70)),
        ((0.46, 0.75), (0.54, 0.75)),
    ]
    return build_template_result("tree_icon", style_name, style, rng, base_lines, finishing_lines, base_count, finishing_count)
