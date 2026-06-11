"""Generate deterministic V1 synthetic stroke programs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from Source.Synthetic.Templates import TEMPLATE_NAMES, TEMPLATES
from Source.Renderer.stroke_schema import DEFAULT_CANVAS_SIZE, STROKE_PROGRAM_VERSION, load_stroke_program_json


DEFAULT_STYLE_CONFIG = Path("Config/stroke_styles.json")
DEFAULT_STROKE_COUNT = 256
DEFAULT_BASE_STROKES = 192
DEFAULT_FINISHING_STROKES = 64
DEFAULT_RANDOM_STYLE_EXCLUDE = {"mono_line"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic BrushWright stroke program.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stroke-count", type=int, default=DEFAULT_STROKE_COUNT)
    parser.add_argument("--base-count", type=int, default=None)
    parser.add_argument("--finishing-count", type=int, default=None)
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--style-config", type=Path, default=DEFAULT_STYLE_CONFIG)
    parser.add_argument("--template", choices=TEMPLATE_NAMES, default=None)
    parser.add_argument("--style", default=None)
    args = parser.parse_args(argv)

    program = generate_program(
        seed=args.seed,
        stroke_count=args.stroke_count,
        base_count=args.base_count,
        finishing_count=args.finishing_count,
        canvas_size=args.canvas_size,
        style_config_path=args.style_config,
        template_name=args.template,
        style_name=args.style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(program, output_file, indent=2)
        output_file.write("\n")
    return 0


def generate_program(
    seed: int,
    stroke_count: int = DEFAULT_STROKE_COUNT,
    base_count: int | None = None,
    finishing_count: int | None = None,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    style_config_path: Path = DEFAULT_STYLE_CONFIG,
    template_name: str | None = None,
    style_name: str | None = None,
) -> dict[str, Any]:
    return generate_program_bundle(
        seed=seed,
        stroke_count=stroke_count,
        base_count=base_count,
        finishing_count=finishing_count,
        canvas_size=canvas_size,
        style_config_path=style_config_path,
        template_name=template_name,
        style_name=style_name,
    )["full_program"]


def generate_program_bundle(
    seed: int,
    stroke_count: int = DEFAULT_STROKE_COUNT,
    base_count: int | None = None,
    finishing_count: int | None = None,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    style_config_path: Path = DEFAULT_STYLE_CONFIG,
    template_name: str | None = None,
    style_name: str | None = None,
) -> dict[str, Any]:
    if stroke_count <= 0:
        raise ValueError("stroke_count must be positive")
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    styles = _load_styles(style_config_path)
    rng = random.Random(seed)
    base_count, finishing_count = _resolve_counts(
        stroke_count=stroke_count,
        base_count=base_count,
        finishing_count=finishing_count,
    )
    selected_template = template_name or rng.choice(TEMPLATE_NAMES)
    selected_style = style_name or _choose_default_style(styles, rng)
    if selected_template not in TEMPLATES:
        raise ValueError(f"unknown template: {selected_template}")
    if selected_style not in styles:
        raise ValueError(f"unknown style: {selected_style}")

    template_result = TEMPLATES[selected_template](
        rng=rng,
        style_name=selected_style,
        style=styles[selected_style],
        base_count=base_count,
        finishing_count=finishing_count,
    )
    base_strokes = template_result["base_strokes"]
    finishing_strokes = template_result["finishing_strokes"]
    full_strokes = base_strokes + finishing_strokes
    metadata = {
        "generator": "Source.Synthetic.generate_programs",
        "seed": seed,
        "stroke_count": len(full_strokes),
        "base_count": len(base_strokes),
        "finishing_count": len(finishing_strokes),
        "style_config": str(style_config_path),
        "style_family": "electronic_simple_icons",
        "template": selected_template,
        "style": selected_style,
    }

    full_program = _program(canvas_size=canvas_size, metadata=metadata, strokes=full_strokes)
    base_program = _program(
        canvas_size=canvas_size,
        metadata={**metadata, "split": "base", "stroke_count": len(base_strokes)},
        strokes=base_strokes,
    )
    finishing_program = _program(
        canvas_size=canvas_size,
        metadata={**metadata, "split": "finishing", "stroke_count": len(finishing_strokes)},
        strokes=finishing_strokes,
    )
    load_stroke_program_json(full_program)
    load_stroke_program_json(base_program)
    load_stroke_program_json(finishing_program)
    return {
        "template": selected_template,
        "style": selected_style,
        "base_strokes": base_program,
        "finishing_strokes": finishing_program,
        "full_program": full_program,
    }


def _program(canvas_size: int, metadata: dict[str, Any], strokes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": STROKE_PROGRAM_VERSION,
        "canvas": {
            "width": canvas_size,
            "height": canvas_size,
        },
        "metadata": metadata,
        "strokes": strokes,
    }


def _resolve_counts(
    stroke_count: int,
    base_count: int | None,
    finishing_count: int | None,
) -> tuple[int, int]:
    if base_count is None and finishing_count is None:
        if stroke_count == DEFAULT_STROKE_COUNT:
            return DEFAULT_BASE_STROKES, DEFAULT_FINISHING_STROKES
        finishing_count = min(DEFAULT_FINISHING_STROKES, max(1, stroke_count // 4))
        base_count = stroke_count - finishing_count
    elif base_count is None:
        if finishing_count is None:
            raise ValueError("finishing_count cannot be None")
        base_count = stroke_count - finishing_count
    elif finishing_count is None:
        finishing_count = stroke_count - base_count

    if base_count <= 0:
        raise ValueError("base_count must be positive")
    if finishing_count <= 0:
        raise ValueError("finishing_count must be positive")
    if base_count + finishing_count != stroke_count:
        raise ValueError("base_count + finishing_count must equal stroke_count")
    return base_count, finishing_count


def _load_styles(style_config_path: Path) -> dict[str, Any]:
    with style_config_path.open("r", encoding="utf-8") as style_file:
        config = json.load(style_file)
    styles = config.get("styles")
    if not isinstance(styles, dict) or not styles:
        raise ValueError("style config must contain non-empty styles object")
    return styles


def _choose_default_style(styles: dict[str, Any], rng: random.Random) -> str:
    candidate_styles = tuple(style_name for style_name in styles if style_name not in DEFAULT_RANDOM_STYLE_EXCLUDE)
    if not candidate_styles:
        candidate_styles = tuple(styles.keys())
    return rng.choice(candidate_styles)


if __name__ == "__main__":
    raise SystemExit(main())
