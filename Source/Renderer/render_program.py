"""Render BrushWright stroke programs with the Paint Transformer renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from Source.PaintTransformerReference.synthesize_samples import render_program_with_paint_transformer
from Source.Renderer.stroke_schema import StrokeSchemaError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a BrushWright stroke program.")
    parser.add_argument("stroke_program", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        render_program_with_paint_transformer(args.stroke_program, args.output_dir)
    except (OSError, RuntimeError, StrokeSchemaError, ValueError) as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

