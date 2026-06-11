"""Generate renderer-scored visual teacher strokes for visual-delta training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

from Source.Model.export_greedy_stroke_optimizer import (
    DEFAULT_ANGLE_DEGREES,
    DEFAULT_ANCHOR_BORDER_MARGIN,
    DEFAULT_ANCHOR_GRID_SIZE,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_BRUSH,
    DEFAULT_DETAIL_CADENCE,
    DEFAULT_DETAIL_MIN_STROKE_MAD_IMPROVEMENT,
    DEFAULT_DETAIL_SIZE_TIERS,
    DEFAULT_DETAIL_START_STROKE,
    DEFAULT_FORCE_MAX_STROKES,
    DEFAULT_MAX_COMPONENT_ANCHORS,
    DEFAULT_MAX_POINT_ANCHORS,
    DEFAULT_MAX_STROKES,
    DEFAULT_MIN_ERROR,
    DEFAULT_MIN_STROKE_MAD_IMPROVEMENT,
    DEFAULT_MIN_STROKE_PIXELS,
    DEFAULT_OPACITIES,
    DEFAULT_OUTSIDE_MASK_PENALTY,
    DEFAULT_SIZE_TIERS,
    DEFAULT_TARGET_MAD_THRESHOLD,
    optimize_greedy_strokes,
)
from Source.Model.export_image_delta_strokes import (
    DEFAULT_TARGET_CONTRACT,
    TARGET_MODE_FINISHED_IMAGE,
    TARGET_MODE_SOURCE_IMAGE,
    TARGET_MODE_TARGET_IMAGE,
    _select_manifest_samples,
    _write_target_image,
)
from Source.Model.export_test_predictions import _program_like


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_SPLITS = ("Train", "Val", "Test")
DEFAULT_TARGET_MODE = TARGET_MODE_TARGET_IMAGE
TEACHER_STROKES_FILE = "visual_teacher_strokes.json"
TEACHER_MANIFEST_FILE = "visual_teacher_manifest.json"
TEACHER_SOURCE = "greedy_stroke_optimizer_v1"


@dataclass(frozen=True)
class VisualTeacherConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    splits: tuple[str, ...] = DEFAULT_SPLITS
    limit: int | None = None
    sample_ids: tuple[str, ...] = ()
    target_mode: str = DEFAULT_TARGET_MODE
    require_target_contract: str | None = DEFAULT_TARGET_CONTRACT
    size_tiers: tuple[int, ...] = DEFAULT_SIZE_TIERS
    detail_size_tiers: tuple[int, ...] = DEFAULT_DETAIL_SIZE_TIERS
    angle_degrees: tuple[float, ...] = DEFAULT_ANGLE_DEGREES
    opacities: tuple[float, ...] = DEFAULT_OPACITIES
    max_strokes: int = DEFAULT_MAX_STROKES
    min_error: float = DEFAULT_MIN_ERROR
    min_stroke_mad_improvement: float = DEFAULT_MIN_STROKE_MAD_IMPROVEMENT
    detail_min_stroke_mad_improvement: float = DEFAULT_DETAIL_MIN_STROKE_MAD_IMPROVEMENT
    target_mad_threshold: float = DEFAULT_TARGET_MAD_THRESHOLD
    detail_start_stroke: int = DEFAULT_DETAIL_START_STROKE
    detail_cadence: int = DEFAULT_DETAIL_CADENCE
    force_max_strokes: bool = DEFAULT_FORCE_MAX_STROKES
    max_component_anchors: int = DEFAULT_MAX_COMPONENT_ANCHORS
    max_point_anchors: int = DEFAULT_MAX_POINT_ANCHORS
    anchor_grid_size: int = DEFAULT_ANCHOR_GRID_SIZE
    anchor_border_margin: int = DEFAULT_ANCHOR_BORDER_MARGIN
    aspect_ratio: float = DEFAULT_ASPECT_RATIO
    min_stroke_pixels: int = DEFAULT_MIN_STROKE_PIXELS
    outside_mask_penalty: float = DEFAULT_OUTSIDE_MASK_PENALTY
    brush: str = DEFAULT_BRUSH


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = generate_visual_teacher_strokes(
        VisualTeacherConfig(
            data_root=args.data_root,
            splits=tuple(args.split or DEFAULT_SPLITS),
            limit=args.limit,
            sample_ids=tuple(args.sample_id or ()),
            target_mode=args.target_mode,
            require_target_contract=args.require_target_contract,
            size_tiers=tuple(args.size_tier or DEFAULT_SIZE_TIERS),
            detail_size_tiers=tuple(args.detail_size_tier or DEFAULT_DETAIL_SIZE_TIERS),
            angle_degrees=tuple(args.angle_degrees or DEFAULT_ANGLE_DEGREES),
            opacities=tuple(args.opacity or DEFAULT_OPACITIES),
            max_strokes=args.max_strokes,
            min_error=args.min_error,
            min_stroke_mad_improvement=args.min_stroke_mad_improvement,
            detail_min_stroke_mad_improvement=args.detail_min_stroke_mad_improvement,
            target_mad_threshold=args.target_mad_threshold,
            detail_start_stroke=args.detail_start_stroke,
            detail_cadence=args.detail_cadence,
            force_max_strokes=args.force_max_strokes,
            max_component_anchors=args.max_component_anchors,
            max_point_anchors=args.max_point_anchors,
            anchor_grid_size=args.anchor_grid_size,
            anchor_border_margin=args.anchor_border_margin,
            aspect_ratio=args.aspect_ratio,
            min_stroke_pixels=args.min_stroke_pixels,
            outside_mask_penalty=args.outside_mask_penalty,
            brush=args.brush,
        )
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def generate_visual_teacher_strokes(config: VisualTeacherConfig) -> dict[str, Any]:
    _validate_config(config)
    split_summaries = []
    total_samples = 0
    total_strokes = 0
    improved_samples = 0
    for split in config.splits:
        split_summary = _generate_split(config, split)
        split_summaries.append(split_summary)
        total_samples += int(split_summary["sample_count"])
        total_strokes += int(split_summary["teacher_stroke_count"])
        improved_samples += int(split_summary["improved_count"])
    summary = {
        "version": 1,
        "teacher_source": TEACHER_SOURCE,
        "target_mode": config.target_mode,
        "data_root": str(config.data_root),
        "splits": split_summaries,
        "sample_count": total_samples,
        "improved_count": improved_samples,
        "teacher_stroke_count": total_strokes,
        "visual_improvement_rate": improved_samples / total_samples if total_samples else 0.0,
    }
    _write_json(config.data_root / "visual_teacher_manifest.json", summary)
    return summary


def _generate_split(config: VisualTeacherConfig, split: str) -> dict[str, Any]:
    split_root = config.data_root / split
    manifest_path = split_root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    if config.require_target_contract and manifest.get("target_contract") != config.require_target_contract:
        raise ValueError(f"{manifest_path} target_contract must be {config.require_target_contract!r}")
    limit = config.limit if config.limit is not None else len(manifest.get("samples", []))
    samples = _select_manifest_samples(manifest.get("samples", []), config.sample_ids, limit)
    entries = []
    with tempfile.TemporaryDirectory(prefix="brushwright_teacher_") as temp_name:
        temp_root = Path(temp_name)
        for index, sample_entry in enumerate(samples, start=1):
            sample_dir = split_root / sample_entry["path"]
            sample = _read_json(sample_dir / "sample.json")
            if config.require_target_contract and sample.get("target_contract") != config.require_target_contract:
                raise ValueError(
                    f"{sample_dir / 'sample.json'} target_contract must be {config.require_target_contract!r}"
                )
            print(f"[{split} {index}/{len(samples)}] generating teacher strokes for {sample['sample_id']}", flush=True)
            entries.append(_generate_sample_teacher(config, sample_dir, sample, temp_root))

    manifest["target_strokes_source"] = TEACHER_SOURCE
    manifest["target_strokes_file"] = TEACHER_STROKES_FILE
    manifest["target_strokes_manifest"] = TEACHER_MANIFEST_FILE
    _write_json(manifest_path, manifest)
    improved = [entry for entry in entries if entry["estimated_mad_improvement"] > 0.0]
    summary = {
        "split": split,
        "sample_count": len(entries),
        "improved_count": len(improved),
        "teacher_stroke_count": sum(int(entry["teacher_stroke_count"]) for entry in entries),
        "mean_estimated_mad_improvement": (
            sum(float(entry["estimated_mad_improvement"]) for entry in entries) / len(entries) if entries else 0.0
        ),
        "samples": entries,
    }
    _write_json(split_root / "visual_teacher_manifest.json", summary)
    return summary


def _generate_sample_teacher(
    config: VisualTeacherConfig,
    sample_dir: Path,
    sample: dict[str, Any],
    temp_root: Path,
) -> dict[str, Any]:
    target_path = _teacher_target_path(config, sample_dir, sample, temp_root)
    strokes, greedy_manifest, _ = optimize_greedy_strokes(
        sample_dir / sample["draft_image"],
        target_path,
        size_tiers=config.size_tiers,
        detail_size_tiers=config.detail_size_tiers,
        angle_degrees=config.angle_degrees,
        opacities=config.opacities,
        max_strokes=config.max_strokes,
        min_error=config.min_error,
        min_stroke_mad_improvement=config.min_stroke_mad_improvement,
        detail_min_stroke_mad_improvement=config.detail_min_stroke_mad_improvement,
        target_mad_threshold=config.target_mad_threshold,
        detail_start_stroke=config.detail_start_stroke,
        detail_cadence=config.detail_cadence,
        force_max_strokes=config.force_max_strokes,
        max_component_anchors=config.max_component_anchors,
        max_point_anchors=config.max_point_anchors,
        anchor_grid_size=config.anchor_grid_size,
        anchor_border_margin=config.anchor_border_margin,
        aspect_ratio=config.aspect_ratio,
        min_stroke_pixels=config.min_stroke_pixels,
        outside_mask_penalty=config.outside_mask_penalty,
        brush=config.brush,
    )
    fallback_program = _read_json(sample_dir / sample["finishing_strokes"])
    teacher_program = _program_like(
        fallback_program,
        {
            **dict(fallback_program.get("metadata", {})),
            "target_strokes_source": TEACHER_SOURCE,
            "target_mode": config.target_mode,
            "sample_id": sample["sample_id"],
            "split": "visual_teacher_strokes",
        },
        strokes,
    )
    teacher_manifest = {
        "version": 1,
        "teacher_source": TEACHER_SOURCE,
        "target_mode": config.target_mode,
        "teacher_stroke_count": len(strokes),
        "initial_mad": greedy_manifest["initial_mad"],
        "final_mad_estimate": greedy_manifest["final_mad_estimate"],
        "estimated_mad_improvement": greedy_manifest["estimated_mad_improvement"],
        "stop_reason": greedy_manifest["stop_reason"],
        "total_candidates_scored": greedy_manifest["total_candidates_scored"],
        "optimizer": {
            "max_strokes": config.max_strokes,
            "size_tiers": list(config.size_tiers),
            "detail_size_tiers": list(config.detail_size_tiers),
            "min_stroke_mad_improvement": config.min_stroke_mad_improvement,
            "detail_min_stroke_mad_improvement": config.detail_min_stroke_mad_improvement,
            "target_mad_threshold": config.target_mad_threshold,
        },
    }
    _write_json(sample_dir / TEACHER_STROKES_FILE, teacher_program)
    _write_json(sample_dir / TEACHER_MANIFEST_FILE, teacher_manifest)
    updated_sample = dict(sample)
    updated_sample["visual_teacher_strokes"] = TEACHER_STROKES_FILE
    updated_sample["visual_teacher_manifest"] = TEACHER_MANIFEST_FILE
    updated_sample["target_strokes_source"] = TEACHER_SOURCE
    updated_sample["target_strokes_target_mode"] = config.target_mode
    _write_json(sample_dir / "sample.json", updated_sample)
    return {
        "sample_id": sample["sample_id"],
        "path": sample_dir.name,
        "teacher_stroke_count": len(strokes),
        "estimated_mad_improvement": greedy_manifest["estimated_mad_improvement"],
        "initial_mad": greedy_manifest["initial_mad"],
        "final_mad_estimate": greedy_manifest["final_mad_estimate"],
        "stop_reason": greedy_manifest["stop_reason"],
    }


def _teacher_target_path(
    config: VisualTeacherConfig,
    sample_dir: Path,
    sample: dict[str, Any],
    temp_root: Path,
) -> Path:
    if config.target_mode == TARGET_MODE_TARGET_IMAGE:
        target_name = sample.get("target_image")
        if target_name:
            target_path = sample_dir / str(target_name)
            if not target_path.exists():
                raise ValueError(f"target image does not exist: {target_path}")
            return target_path
    output_dir = temp_root / str(sample["sample_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return _write_target_image(sample_dir, sample, output_dir, config.target_mode)


def _validate_config(config: VisualTeacherConfig) -> None:
    if not config.splits:
        raise ValueError("splits must not be empty")
    if config.limit is not None and config.limit <= 0 and not config.sample_ids:
        raise ValueError("limit must be positive unless sample_ids are provided")
    if config.target_mode not in (TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE):
        raise ValueError(f"unsupported target_mode: {config.target_mode}")
    if config.max_strokes <= 0:
        raise ValueError("max_strokes must be positive")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate visual teacher strokes for visual-delta training.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--target-mode", choices=(TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE), default=DEFAULT_TARGET_MODE)
    parser.add_argument("--require-target-contract", default=DEFAULT_TARGET_CONTRACT)
    parser.add_argument("--size-tier", type=int, action="append", default=None)
    parser.add_argument("--detail-size-tier", type=int, action="append", default=None)
    parser.add_argument("--angle-degrees", type=float, action="append", default=None)
    parser.add_argument("--opacity", type=float, action="append", default=None)
    parser.add_argument("--max-strokes", type=int, default=DEFAULT_MAX_STROKES)
    parser.add_argument("--min-error", type=float, default=DEFAULT_MIN_ERROR)
    parser.add_argument("--min-stroke-mad-improvement", type=float, default=DEFAULT_MIN_STROKE_MAD_IMPROVEMENT)
    parser.add_argument("--detail-min-stroke-mad-improvement", type=float, default=DEFAULT_DETAIL_MIN_STROKE_MAD_IMPROVEMENT)
    parser.add_argument("--target-mad-threshold", type=float, default=DEFAULT_TARGET_MAD_THRESHOLD)
    parser.add_argument("--detail-start-stroke", type=int, default=DEFAULT_DETAIL_START_STROKE)
    parser.add_argument("--detail-cadence", type=int, default=DEFAULT_DETAIL_CADENCE)
    parser.add_argument("--force-max-strokes", action="store_true")
    parser.add_argument("--max-component-anchors", type=int, default=DEFAULT_MAX_COMPONENT_ANCHORS)
    parser.add_argument("--max-point-anchors", type=int, default=DEFAULT_MAX_POINT_ANCHORS)
    parser.add_argument("--anchor-grid-size", type=int, default=DEFAULT_ANCHOR_GRID_SIZE)
    parser.add_argument("--anchor-border-margin", type=int, default=DEFAULT_ANCHOR_BORDER_MARGIN)
    parser.add_argument("--aspect-ratio", type=float, default=DEFAULT_ASPECT_RATIO)
    parser.add_argument("--min-stroke-pixels", type=int, default=DEFAULT_MIN_STROKE_PIXELS)
    parser.add_argument("--outside-mask-penalty", type=float, default=DEFAULT_OUTSIDE_MASK_PENALTY)
    parser.add_argument("--brush", default=DEFAULT_BRUSH)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
