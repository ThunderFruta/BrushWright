"""Split full stroke programs into base and finishing stroke programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from Source.Renderer.stroke_schema import STROKE_PROGRAM_VERSION, load_stroke_program_json


DEFAULT_FINISHING_STROKES = 64


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a full stroke program into base and finishing programs.")
    parser.add_argument("full_program", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-count", type=int, default=None)
    parser.add_argument("--finishing-count", type=int, default=DEFAULT_FINISHING_STROKES)
    args = parser.parse_args(argv)

    with args.full_program.open("r", encoding="utf-8") as input_file:
        full_program = json.load(input_file)
    split = split_program(
        full_program=full_program,
        base_count=args.base_count,
        finishing_count=args.finishing_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "base_strokes.json", split["base_strokes"])
    _write_json(args.output_dir / "finishing_strokes.json", split["finishing_strokes"])
    _write_json(args.output_dir / "split_manifest.json", split["manifest"])
    return 0


def split_program(
    full_program: dict[str, Any],
    base_count: int | None = None,
    finishing_count: int = DEFAULT_FINISHING_STROKES,
) -> dict[str, Any]:
    program = load_stroke_program_json(full_program)
    total_strokes = len(program.strokes)
    if finishing_count <= 0:
        raise ValueError("finishing_count must be positive")
    if finishing_count >= total_strokes:
        raise ValueError("finishing_count must be smaller than total stroke count")
    if base_count is None:
        base_count = total_strokes - finishing_count
    if base_count <= 0:
        raise ValueError("base_count must be positive")
    if base_count + finishing_count != total_strokes:
        raise ValueError("base_count + finishing_count must equal total stroke count")

    base_strokes = list(full_program["strokes"][:base_count])
    finishing_start = base_count
    finishing_end = base_count + finishing_count
    finishing_strokes = list(full_program["strokes"][finishing_start:finishing_end])

    shared_metadata = dict(full_program.get("metadata", {}))
    base_program = _program_like(full_program, base_strokes, shared_metadata, "base")
    finishing_program = _program_like(full_program, finishing_strokes, shared_metadata, "finishing")
    manifest = {
        "version": STROKE_PROGRAM_VERSION,
        "total_strokes": total_strokes,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "withheld_start_index": finishing_start,
        "withheld_end_index_exclusive": finishing_end,
    }
    load_stroke_program_json(base_program)
    load_stroke_program_json(finishing_program)
    return {
        "base_strokes": base_program,
        "finishing_strokes": finishing_program,
        "manifest": manifest,
    }


def _program_like(
    full_program: dict[str, Any],
    strokes: list[dict[str, Any]],
    metadata: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    next_metadata = dict(metadata)
    next_metadata["split"] = split_name
    next_metadata["stroke_count"] = len(strokes)
    return {
        "version": STROKE_PROGRAM_VERSION,
        "canvas": full_program.get("canvas", {}),
        "metadata": next_metadata,
        "strokes": strokes,
    }


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
