from __future__ import annotations

import random
from typing import Any

from Source.Synthetic.Templates.template_utils import build_template_result, box_lines


def generate_house_icon(
    rng: random.Random,
    style_name: str,
    style: dict[str, Any],
    base_count: int,
    finishing_count: int,
) -> dict[str, Any]:
    base_lines = [
        ((0.24, 0.48), (0.50, 0.24)),
        ((0.50, 0.24), (0.76, 0.48)),
        ((0.30, 0.48), (0.30, 0.80)),
        ((0.70, 0.48), (0.70, 0.80)),
        ((0.30, 0.80), (0.70, 0.80)),
        ((0.28, 0.48), (0.72, 0.48)),
        ((0.42, 0.80), (0.42, 0.62)),
        ((0.58, 0.62), (0.58, 0.80)),
        ((0.42, 0.62), (0.58, 0.62)),
    ]
    finishing_lines = [
        *box_lines(0.36, 0.54, 0.47, 0.64),
        *box_lines(0.55, 0.54, 0.66, 0.64),
        ((0.415, 0.59), (0.475, 0.59)),
        ((0.415, 0.54), (0.415, 0.64)),
        ((0.55, 0.59), (0.66, 0.59)),
        ((0.605, 0.54), (0.605, 0.64)),
        ((0.59, 0.72), (0.62, 0.72)),
        ((0.34, 0.46), (0.50, 0.31)),
        ((0.50, 0.31), (0.66, 0.46)),
    ]
    return build_template_result("house_icon", style_name, style, rng, base_lines, finishing_lines, base_count, finishing_count)

