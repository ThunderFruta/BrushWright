from __future__ import annotations

import random
from typing import Any

from Source.Synthetic.Templates.template_utils import build_template_result, box_lines, radial_lines, regular_polygon_lines


def generate_geometric_badge(
    rng: random.Random,
    style_name: str,
    style: dict[str, Any],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    base_lines = [
        *regular_polygon_lines(0.50, 0.50, 0.34, 6, rotation=0.52),
        *box_lines(0.34, 0.34, 0.66, 0.66),
        ((0.34, 0.34), (0.66, 0.66)),
        ((0.66, 0.34), (0.34, 0.66)),
    ]
    finishing_lines = [
        *regular_polygon_lines(0.50, 0.50, 0.20, 6, rotation=0.0),
        *radial_lines(0.50, 0.50, 0.08, 0.30, 8, rotation=0.15),
        ((0.40, 0.50), (0.60, 0.50)),
        ((0.50, 0.40), (0.50, 0.60)),
        ((0.43, 0.43), (0.57, 0.57)),
        ((0.57, 0.43), (0.43, 0.57)),
    ]
    return build_template_result("geometric_badge", style_name, style, rng, base_lines, finishing_lines, base_count, finishing_count)
