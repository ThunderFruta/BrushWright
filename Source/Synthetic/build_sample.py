"""Build a complete pre-ML supervised BrushWright sample."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from Source.Synthetic.generate_programs import DEFAULT_FINISHING_STROKES, DEFAULT_STROKE_COUNT, generate_program_bundle


DEFAULT_BASE_STROKES = 192
DEFAULT_GENERATED_ROOT = Path("Outputs/Samples")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a complete synthetic BrushWright sample.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stroke-count", type=int, default=DEFAULT_STROKE_COUNT)
    parser.add_argument("--base-count", type=int, default=DEFAULT_BASE_STROKES)
    parser.add_argument("--finishing-count", type=int, default=DEFAULT_FINISHING_STROKES)
    parser.add_argument("--template", default=None)
    parser.add_argument("--style", default=None)
    args = parser.parse_args(argv)

    build_sample(
        seed=args.seed,
        output_dir=args.output_dir or _default_output_dir(args.seed),
        stroke_count=args.stroke_count,
        base_count=args.base_count,
        finishing_count=args.finishing_count,
        template_name=args.template,
        style_name=args.style,
    )
    return 0


def _default_output_dir(seed: int) -> Path:
    return DEFAULT_GENERATED_ROOT / f"sample_{seed:06d}"


def build_sample(
    seed: int,
    output_dir: Path,
    stroke_count: int = DEFAULT_STROKE_COUNT,
    base_count: int = DEFAULT_BASE_STROKES,
    finishing_count: int = DEFAULT_FINISHING_STROKES,
    template_name: str | None = None,
    style_name: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = generate_program_bundle(
        seed=seed,
        stroke_count=stroke_count,
        base_count=base_count,
        finishing_count=finishing_count,
        template_name=template_name,
        style_name=style_name,
    )
    full_program = bundle["full_program"]
    base_program = bundle["base_strokes"]
    finishing_program = bundle["finishing_strokes"]
    split_manifest = {
        "version": 1,
        "method": "template_roles",
        "template": bundle["template"],
        "style": bundle["style"],
        "total_strokes": len(full_program["strokes"]),
        "base_count": len(base_program["strokes"]),
        "finishing_count": len(finishing_program["strokes"]),
        "withheld_start_index": len(base_program["strokes"]),
        "withheld_end_index_exclusive": len(full_program["strokes"]),
    }

    full_program_path = output_dir / "full_program.json"
    base_path = output_dir / "base_strokes.json"
    finishing_path = output_dir / "finishing_strokes.json"
    split_manifest_path = output_dir / "split_manifest.json"

    _write_json(full_program_path, full_program)
    _write_json(base_path, base_program)
    _write_json(finishing_path, finishing_program)
    _write_json(split_manifest_path, split_manifest)

    draft_render_dir = output_dir / "draft_render"
    finished_render_dir = output_dir / "finished_render"
    _run_renderer(base_path, draft_render_dir)
    _run_renderer(full_program_path, finished_render_dir)

    draft_image = output_dir / "draft.png"
    finished_image = output_dir / "finished.png"
    shutil.copyfile(draft_render_dir / "final.png", draft_image)
    shutil.copyfile(finished_render_dir / "final.png", finished_image)

    sample = {
        "version": 1,
        "sample_id": f"sample_{seed:06d}",
        "seed": seed,
        "canvas": full_program["canvas"],
        "stroke_count": stroke_count,
        "base_count": len(base_program["strokes"]),
        "finishing_count": len(finishing_program["strokes"]),
        "template": bundle["template"],
        "style": bundle["style"],
        "full_program": _relative_to_sample(output_dir, full_program_path),
        "base_strokes": _relative_to_sample(output_dir, base_path),
        "finishing_strokes": _relative_to_sample(output_dir, finishing_path),
        "draft_image": _relative_to_sample(output_dir, draft_image),
        "finished_image": _relative_to_sample(output_dir, finished_image),
        "draft_render_manifest": _relative_to_sample(output_dir, draft_render_dir / "render_manifest.json"),
        "finished_render_manifest": _relative_to_sample(output_dir, finished_render_dir / "render_manifest.json"),
        "split_manifest": _relative_to_sample(output_dir, split_manifest_path),
    }
    _write_json(output_dir / "sample.json", sample)
    return sample


def _run_renderer(stroke_program: Path, output_dir: Path) -> None:
    root_dir = Path(__file__).resolve().parents[2]
    subprocess.run(
        [str(root_dir / "Scripts" / "run_renderer.sh"), str(stroke_program), str(output_dir)],
        cwd=root_dir,
        check=True,
    )


def _relative_to_sample(sample_dir: Path, path: Path) -> str:
    return str(path.relative_to(sample_dir))


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
