from __future__ import annotations

import random
from typing import Any

from Source.Synthetic.Templates.template_utils import build_template_result, radial_lines, regular_polygon_lines


def generate_flower_icon(
    rng: random.Random,
    style_name: str,
    style: dict[str, Any],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    base_lines = [
        *radial_lines(0.50, 0.43, 0.07, 0.24, 8, rotation=0.2),
        ((0.50, 0.50), (0.50, 0.82)),
        ((0.50, 0.66), (0.36, 0.58)),
        ((0.50, 0.70), (0.64, 0.61)),
        ((0.36, 0.58), (0.43, 0.64)),
        ((0.64, 0.61), (0.57, 0.67)),
    ]
    finishing_lines = [
        *regular_polygon_lines(0.50, 0.43, 0.07, 8, rotation=0.1),
        *radial_lines(0.50, 0.43, 0.12, 0.20, 8, rotation=0.6),
        ((0.47, 0.60), (0.53, 0.60)),
        ((0.47, 0.68), (0.53, 0.68)),
        ((0.47, 0.76), (0.53, 0.76)),
    ]
    return build_template_result("flower_icon", style_name, style, rng, base_lines, finishing_lines, base_count, finishing_count)

