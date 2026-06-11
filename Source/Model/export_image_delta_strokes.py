"""Export deterministic image-delta stroke predictions.

This is a renderer-first baseline for the visual-delta task. It compiles the
visible draft-to-target delta into editable BrushWright strokes by placing
target-colored Paint Transformer rectangle strokes over changed image cells.
It is not an image-to-image output path: the artifact remains added_strokes.json
and is rendered through the Paint Transformer adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from Source.Model.export_test_predictions import _program_like, _write_comparison_strip
from Source.Model.prediction_diagnostics import compute_prediction_diagnostics
from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/ImageDeltaStrokeCompilerV1")
DEFAULT_SPLIT = "Test"
DEFAULT_LIMIT = 4
DEFAULT_CELL_SIZE = 20
DEFAULT_STRIDE = 14
DEFAULT_MAX_STROKES = 512
DEFAULT_MIN_ERROR = 0.025
DEFAULT_MIN_CELL_CHANGED_PIXELS = 4
DEFAULT_MIN_STROKE_PIXELS = 4
DEFAULT_STROKE_SCALE = 0.9
DEFAULT_ASPECT_RATIO = 1.4
DEFAULT_OPACITY = 0.70
DEFAULT_BRUSH = "paint_transformer_rect"
DEFAULT_TARGET_CONTRACT = "paint_transformer_original_image_target_v1"
DEFAULT_MIN_CHANGED_PIXEL_RATIO = 0.01
DEFAULT_MIN_GRADIENT_IMPROVEMENT = 0.0
DEFAULT_MIN_EDGE_OVERLAP = 0.02
TARGET_MODE_FINISHED_IMAGE = "finished-image"
TARGET_MODE_TARGET_IMAGE = "target-image"
TARGET_MODE_SOURCE_IMAGE = "source-image"
DEFAULT_TARGET_MODE = TARGET_MODE_TARGET_IMAGE
DEFAULT_RECURSIVE_PASSES = 1
DEFAULT_STOP_ON_NON_IMPROVEMENT = True
DEFAULT_MIN_PASS_MAD_IMPROVEMENT = 0.10
DEFAULT_TARGET_MAD_THRESHOLD = 3.0


@dataclass(frozen=True)
class ImageDeltaStrokeConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    limit: int = DEFAULT_LIMIT
    sample_ids: tuple[str, ...] = ()
    cell_size: int = DEFAULT_CELL_SIZE
    stride: int = DEFAULT_STRIDE
    max_strokes: int = DEFAULT_MAX_STROKES
    min_error: float = DEFAULT_MIN_ERROR
    min_cell_changed_pixels: int = DEFAULT_MIN_CELL_CHANGED_PIXELS
    min_stroke_pixels: int = DEFAULT_MIN_STROKE_PIXELS
    stroke_scale: float = DEFAULT_STROKE_SCALE
    aspect_ratio: float = DEFAULT_ASPECT_RATIO
    opacity: float = DEFAULT_OPACITY
    brush: str = DEFAULT_BRUSH
    require_target_contract: str | None = DEFAULT_TARGET_CONTRACT
    min_changed_pixel_ratio: float = DEFAULT_MIN_CHANGED_PIXEL_RATIO
    min_gradient_improvement: float = DEFAULT_MIN_GRADIENT_IMPROVEMENT
    min_edge_overlap: float = DEFAULT_MIN_EDGE_OVERLAP
    target_mode: str = DEFAULT_TARGET_MODE
    recursive_passes: int = DEFAULT_RECURSIVE_PASSES
    stop_on_non_improvement: bool = DEFAULT_STOP_ON_NON_IMPROVEMENT
    min_pass_mad_improvement: float = DEFAULT_MIN_PASS_MAD_IMPROVEMENT
    target_mad_threshold: float = DEFAULT_TARGET_MAD_THRESHOLD
    render: bool = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exported = export_image_delta_strokes(
        ImageDeltaStrokeConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            split=args.split,
            limit=args.limit,
            sample_ids=tuple(args.sample_id or ()),
            cell_size=args.cell_size,
            stride=args.stride,
            max_strokes=args.max_strokes,
            min_error=args.min_error,
            min_cell_changed_pixels=args.min_cell_changed_pixels,
            min_stroke_pixels=args.min_stroke_pixels,
            stroke_scale=args.stroke_scale,
            aspect_ratio=args.aspect_ratio,
            opacity=args.opacity,
            brush=args.brush,
            require_target_contract=args.require_target_contract,
            min_changed_pixel_ratio=args.min_changed_pixel_ratio,
            min_gradient_improvement=args.min_gradient_improvement,
            min_edge_overlap=args.min_edge_overlap,
            target_mode=args.target_mode,
            recursive_passes=args.recursive_passes,
            stop_on_non_improvement=args.stop_on_non_improvement,
            min_pass_mad_improvement=args.min_pass_mad_improvement,
            target_mad_threshold=args.target_mad_threshold,
            render=not args.no_render,
        )
    )
    print(json.dumps({"exported": exported}, indent=2), flush=True)
    return 0


def export_image_delta_strokes(config: ImageDeltaStrokeConfig) -> list[dict[str, Any]]:
    _validate_config(config)
    split_root = config.data_root / config.split
    manifest = _read_json(split_root / "dataset_manifest.json")
    if config.require_target_contract and manifest.get("target_contract") != config.require_target_contract:
        raise ValueError(
            f"{split_root / 'dataset_manifest.json'} target_contract must be "
            f"{config.require_target_contract!r}"
        )

    samples = _select_manifest_samples(manifest.get("samples", []), config.sample_ids, config.limit)
    prepared_root = prepare_latest_output_root(config.output_root)
    output_root = prepared_root / config.split
    output_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, sample_entry in enumerate(samples, start=1):
        sample_dir = split_root / sample_entry["path"]
        sample = _read_json(sample_dir / "sample.json")
        if config.require_target_contract and sample.get("target_contract") != config.require_target_contract:
            raise ValueError(
                f"{sample_dir / 'sample.json'} target_contract must be {config.require_target_contract!r}"
            )
        output_dir = output_root / str(sample["sample_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(samples)}] compiling {sample['sample_id']} -> {output_dir}", flush=True)
        exported.append(_export_sample(sample_dir, sample, output_dir, config))

    _write_json(
        prepared_root / "export_manifest.json",
        {"version": 1, "split": config.split, "summary": _export_summary(exported), "samples": exported},
    )
    return exported


def compile_image_delta_strokes(
    draft_path: Path,
    target_path: Path,
    *,
    cell_size: int = DEFAULT_CELL_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_strokes: int = DEFAULT_MAX_STROKES,
    min_error: float = DEFAULT_MIN_ERROR,
    min_cell_changed_pixels: int = DEFAULT_MIN_CELL_CHANGED_PIXELS,
    min_stroke_pixels: int = DEFAULT_MIN_STROKE_PIXELS,
    stroke_scale: float = DEFAULT_STROKE_SCALE,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    opacity: float = DEFAULT_OPACITY,
    brush: str = DEFAULT_BRUSH,
) -> list[dict[str, Any]]:
    draft, target = _load_image_pair(draft_path, target_path)
    height, width, _ = draft.shape
    error = np.max(np.abs(target - draft), axis=2)
    edit_mask = error > min_error
    candidates = []
    order = 0
    for top in _offsets(height, cell_size, stride):
        for left in _offsets(width, cell_size, stride):
            bottom = min(height, top + cell_size)
            right = min(width, left + cell_size)
            mask_patch = edit_mask[top:bottom, left:right]
            changed_count = int(mask_patch.sum())
            if changed_count < min_cell_changed_pixels:
                continue
            error_patch = error[top:bottom, left:right]
            target_patch = target[top:bottom, left:right]
            ys, xs = np.nonzero(mask_patch)
            weights = error_patch[mask_patch].astype(np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            center_x = left + float(np.average(xs + 0.5, weights=weights))
            center_y = top + float(np.average(ys + 0.5, weights=weights))
            color = np.average(target_patch[mask_patch], axis=0, weights=weights)
            orientation = _weighted_patch_orientation(xs, ys, weights)
            major_pixels, minor_pixels = _weighted_patch_extent(
                xs,
                ys,
                weights,
                orientation_radians=orientation,
                min_stroke_pixels=float(min_stroke_pixels),
                cell_size=float(cell_size),
                stroke_scale=stroke_scale,
                aspect_ratio=aspect_ratio,
            )
            length_pixels = min(width, major_pixels)
            width_pixels = min(height, minor_pixels)
            mean_error = float(error_patch[mask_patch].mean())
            max_error = float(error_patch[mask_patch].max())
            score = mean_error * math.sqrt(changed_count) + 0.25 * max_error
            candidates.append(
                {
                    "stroke": {
                        "x": _clamp(center_x / width),
                        "y": _clamp(center_y / height),
                        "angle": _brushwright_angle(orientation),
                        "length": _clamp(length_pixels / width),
                        "width": _clamp(width_pixels / height),
                        "color": [_clamp(float(channel)) for channel in color.tolist()],
                        "opacity": _clamp(opacity),
                        "brush": brush,
                    },
                    "score": score,
                    "order": order,
                }
            )
            order += 1
    candidates.sort(key=lambda entry: (float(entry["score"]), -int(entry["order"])), reverse=True)
    selected = candidates[:max_strokes]
    # Lower-score strokes are rendered first so high-error cells can land on top.
    selected.sort(key=lambda entry: (float(entry["score"]), -int(entry["order"])))
    return [dict(entry["stroke"]) for entry in selected]


def _export_sample(
    sample_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    config: ImageDeltaStrokeConfig,
) -> dict[str, Any]:
    if config.recursive_passes > 1:
        return _export_recursive_sample(sample_dir, sample, output_dir, config)

    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    draft_path = sample_dir / sample["draft_image"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(draft_path, output_dir / "draft.png")
    target_path = _write_target_image(sample_dir, sample, output_dir, config.target_mode)
    predicted_strokes = compile_image_delta_strokes(
        output_dir / "draft.png",
        target_path,
        cell_size=config.cell_size,
        stride=config.stride,
        max_strokes=config.max_strokes,
        min_error=config.min_error,
        min_cell_changed_pixels=config.min_cell_changed_pixels,
        min_stroke_pixels=config.min_stroke_pixels,
        stroke_scale=config.stroke_scale,
        aspect_ratio=config.aspect_ratio,
        opacity=config.opacity,
        brush=config.brush,
    )
    metadata = {
        **dict(target_program.get("metadata", {})),
        "prediction_source": "image_delta_stroke_compiler_v1",
        "sample_id": sample["sample_id"],
        "split": "image_delta_added_strokes",
        "target_mode": config.target_mode,
        "compiler": {
            "cell_size": config.cell_size,
            "stride": config.stride,
            "max_strokes": config.max_strokes,
            "min_error": config.min_error,
            "min_cell_changed_pixels": config.min_cell_changed_pixels,
            "min_stroke_pixels": config.min_stroke_pixels,
            "stroke_scale": config.stroke_scale,
            "aspect_ratio": config.aspect_ratio,
            "selected_count": len(predicted_strokes),
        },
    }
    predicted_finishing_program = _program_like(target_program, metadata, predicted_strokes)
    predicted_full_program = _program_like(
        base_program,
        {
            **dict(base_program.get("metadata", {})),
            "prediction_source": "image_delta_stroke_compiler_v1",
            "sample_id": sample["sample_id"],
            "split": "base_plus_image_delta_strokes",
        },
        [*base_program["strokes"], *predicted_strokes],
    )

    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(
        output_dir / "sample.json",
        sample | {"prediction_source": "image_delta_stroke_compiler_v1", "prediction_target_mode": config.target_mode},
    )

    if config.render:
        render_program_final_with_paint_transformer(
            output_dir / "added_strokes.json",
            output_dir / "predicted.png",
            background_path=output_dir / "draft.png",
        )
        _write_comparison_strip(
            output_dir / "draft.png",
            output_dir / "target.png",
            output_dir / "predicted.png",
            output_dir / "comparison.png",
        )
        diagnostics = compute_prediction_diagnostics(
            draft_path=output_dir / "draft.png",
            target_path=output_dir / "target.png",
            predicted_path=output_dir / "predicted.png",
            predicted_strokes=predicted_strokes,
            target_strokes=target_program["strokes"] if config.target_mode == TARGET_MODE_FINISHED_IMAGE else None,
            min_changed_pixel_ratio=config.min_changed_pixel_ratio,
            min_gradient_improvement=config.min_gradient_improvement,
            min_edge_overlap=config.min_edge_overlap,
        )
        _write_json(output_dir / "diagnostics.json", diagnostics)
    else:
        diagnostics = {"status": "not_rendered", "visual_improved": None}

    return {
        "sample_id": sample["sample_id"],
        "output_dir": str(output_dir),
        "draft": str(output_dir / "draft.png"),
        "target": str(output_dir / "target.png"),
        "predicted": str(output_dir / "predicted.png") if config.render else None,
        "comparison": str(output_dir / "comparison.png") if config.render else None,
        "diagnostics": str(output_dir / "diagnostics.json") if config.render else None,
        "status": diagnostics["status"],
        "visual_improved": diagnostics["visual_improved"],
        "added_strokes": str(output_dir / "added_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
        "added_stroke_count": len(predicted_strokes),
    }


def _export_recursive_sample(
    sample_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    config: ImageDeltaStrokeConfig,
) -> dict[str, Any]:
    if not config.render:
        raise ValueError("recursive image-delta export requires rendering")

    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    target_path = _write_target_image(sample_dir, sample, output_dir, config.target_mode)

    current_draft_path = output_dir / "draft.png"
    previous_mad = _mean_absolute_difference(current_draft_path, target_path)
    all_strokes: list[dict[str, Any]] = []
    pass_entries: list[dict[str, Any]] = []
    stop_reason = "max_passes"

    for pass_index in range(1, config.recursive_passes + 1):
        if previous_mad <= config.target_mad_threshold:
            stop_reason = "target_threshold"
            break

        pass_dir = output_dir / f"pass_{pass_index:04d}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_draft_path, pass_dir / "draft.png")
        shutil.copy2(target_path, pass_dir / "target.png")
        pass_strokes = compile_image_delta_strokes(
            pass_dir / "draft.png",
            pass_dir / "target.png",
            cell_size=config.cell_size,
            stride=config.stride,
            max_strokes=config.max_strokes,
            min_error=config.min_error,
            min_cell_changed_pixels=config.min_cell_changed_pixels,
            min_stroke_pixels=config.min_stroke_pixels,
            stroke_scale=config.stroke_scale,
            aspect_ratio=config.aspect_ratio,
            opacity=config.opacity,
            brush=config.brush,
        )
        pass_program = _program_like(
            target_program,
            {
                **dict(target_program.get("metadata", {})),
                "prediction_source": "image_delta_stroke_compiler_v1",
                "sample_id": sample["sample_id"],
                "split": "image_delta_recursive_pass",
                "target_mode": config.target_mode,
                "recursive_pass": pass_index,
            },
            pass_strokes,
        )
        _write_json(pass_dir / "added_strokes.json", pass_program)
        if pass_strokes:
            render_program_final_with_paint_transformer(
                pass_dir / "added_strokes.json",
                pass_dir / "predicted.png",
                background_path=pass_dir / "draft.png",
            )
        else:
            shutil.copy2(pass_dir / "draft.png", pass_dir / "predicted.png")
        _write_comparison_strip(pass_dir / "draft.png", pass_dir / "target.png", pass_dir / "predicted.png", pass_dir / "comparison.png")
        diagnostics = compute_prediction_diagnostics(
            draft_path=pass_dir / "draft.png",
            target_path=pass_dir / "target.png",
            predicted_path=pass_dir / "predicted.png",
            predicted_strokes=pass_strokes,
            target_strokes=target_program["strokes"] if config.target_mode == TARGET_MODE_FINISHED_IMAGE else None,
            min_changed_pixel_ratio=config.min_changed_pixel_ratio,
            min_gradient_improvement=config.min_gradient_improvement,
            min_edge_overlap=config.min_edge_overlap,
        )
        current_mad = diagnostics["image_deltas"]["predicted_to_target"]["mean_absolute_difference"]
        pass_improvement = previous_mad - current_mad
        accepted = bool(pass_strokes) and current_mad < previous_mad
        if config.stop_on_non_improvement and pass_improvement < config.min_pass_mad_improvement:
            accepted = False
            stop_reason = "non_improvement"
        elif not pass_strokes:
            stop_reason = "no_strokes"
        elif current_mad <= config.target_mad_threshold:
            stop_reason = "target_threshold"

        diagnostics["recursive_pass"] = {
            "pass_index": pass_index,
            "previous_predicted_to_target_mad": previous_mad,
            "current_predicted_to_target_mad": current_mad,
            "pass_mad_improvement": pass_improvement,
            "accepted": accepted,
        }
        _write_json(pass_dir / "diagnostics.json", diagnostics)
        pass_entries.append(
            {
                "pass_index": pass_index,
                "output_dir": str(pass_dir),
                "draft": str(pass_dir / "draft.png"),
                "target": str(pass_dir / "target.png"),
                "predicted": str(pass_dir / "predicted.png"),
                "comparison": str(pass_dir / "comparison.png"),
                "diagnostics": str(pass_dir / "diagnostics.json"),
                "added_strokes": str(pass_dir / "added_strokes.json"),
                "status": diagnostics["status"],
                "visual_improved": diagnostics["visual_improved"],
                "stroke_count": len(pass_strokes),
                "previous_predicted_to_target_mad": previous_mad,
                "current_predicted_to_target_mad": current_mad,
                "pass_mad_improvement": pass_improvement,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        all_strokes.extend(pass_strokes)
        current_draft_path = pass_dir / "predicted.png"
        previous_mad = current_mad
        if stop_reason != "max_passes":
            break

    predicted_finishing_program = _program_like(
        target_program,
        {
            **dict(target_program.get("metadata", {})),
            "prediction_source": "image_delta_stroke_compiler_v1",
            "sample_id": sample["sample_id"],
            "split": "image_delta_recursive_added_strokes",
            "target_mode": config.target_mode,
            "recursive_export": {
                "recursive_passes_requested": config.recursive_passes,
                "recursive_passes_completed": len(pass_entries),
                "stop_reason": stop_reason,
                "stop_on_non_improvement": config.stop_on_non_improvement,
                "min_pass_mad_improvement": config.min_pass_mad_improvement,
                "target_mad_threshold": config.target_mad_threshold,
                "strokes_per_pass": config.max_strokes,
            },
        },
        all_strokes,
    )
    predicted_full_program = _program_like(
        base_program,
        {
            **dict(base_program.get("metadata", {})),
            "prediction_source": "image_delta_stroke_compiler_v1",
            "sample_id": sample["sample_id"],
            "split": "base_plus_recursive_image_delta_strokes",
            "target_mode": config.target_mode,
        },
        [*base_program["strokes"], *all_strokes],
    )
    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(
        output_dir / "sample.json",
        sample | {"prediction_source": "image_delta_stroke_compiler_v1", "prediction_target_mode": config.target_mode},
    )
    shutil.copy2(current_draft_path, output_dir / "predicted.png")
    _write_comparison_strip(output_dir / "draft.png", output_dir / "target.png", output_dir / "predicted.png", output_dir / "comparison.png")
    final_diagnostics = compute_prediction_diagnostics(
        draft_path=output_dir / "draft.png",
        target_path=output_dir / "target.png",
        predicted_path=output_dir / "predicted.png",
        predicted_strokes=all_strokes,
        target_strokes=target_program["strokes"] if config.target_mode == TARGET_MODE_FINISHED_IMAGE else None,
        min_changed_pixel_ratio=config.min_changed_pixel_ratio,
        min_gradient_improvement=config.min_gradient_improvement,
        min_edge_overlap=config.min_edge_overlap,
    )
    final_diagnostics["recursive_export"] = {
        "passes": pass_entries,
        "stop_reason": stop_reason,
        "total_added_strokes": len(all_strokes),
        "max_possible_added_strokes": config.recursive_passes * config.max_strokes,
        "target_mode": config.target_mode,
    }
    _write_json(output_dir / "diagnostics.json", final_diagnostics)
    _write_json(
        output_dir / "recursive_manifest.json",
        {
            "version": 1,
            "sample_id": sample["sample_id"],
            "target_mode": config.target_mode,
            "stop_reason": stop_reason,
            "total_added_strokes": len(all_strokes),
            "passes": pass_entries,
        },
    )
    return {
        "sample_id": sample["sample_id"],
        "output_dir": str(output_dir),
        "draft": str(output_dir / "draft.png"),
        "target": str(output_dir / "target.png"),
        "predicted": str(output_dir / "predicted.png"),
        "comparison": str(output_dir / "comparison.png"),
        "diagnostics": str(output_dir / "diagnostics.json"),
        "status": final_diagnostics["status"],
        "visual_improved": final_diagnostics["visual_improved"],
        "added_strokes": str(output_dir / "added_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
        "added_stroke_count": len(all_strokes),
        "recursive_manifest": str(output_dir / "recursive_manifest.json"),
        "recursive_passes_completed": len(pass_entries),
        "recursive_stop_reason": stop_reason,
    }


def _export_summary(exported: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [entry for entry in exported if entry.get("visual_improved") is not None]
    improved = [entry for entry in rendered if entry.get("visual_improved")]
    low_change = [entry for entry in rendered if entry.get("status") == "failed_low_pixel_change"]
    structure_failed = [entry for entry in rendered if entry.get("status") == "failed_structure_noise"]
    visual_pass = bool(improved) and not low_change and not structure_failed
    return {
        "sample_count": len(exported),
        "rendered_count": len(rendered),
        "improved_count": len(improved),
        "visual_improvement_rate": len(improved) / len(rendered) if rendered else 0.0,
        "low_change_count": len(low_change),
        "low_change_rate": len(low_change) / len(rendered) if rendered else 0.0,
        "structure_failed_count": len(structure_failed),
        "structure_failed_rate": len(structure_failed) / len(rendered) if rendered else 0.0,
        "checkpoint_status": "visual_pass" if visual_pass else "visual_failed",
        "status_histogram": _status_histogram(exported),
    }


def _select_manifest_samples(
    samples: list[dict[str, Any]],
    sample_ids: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    if not sample_ids:
        return samples[:limit]
    requested = set(sample_ids)
    selected = [sample for sample in samples if Path(str(sample.get("path", ""))).name in requested]
    missing = requested - {Path(str(sample.get("path", ""))).name for sample in selected}
    if missing:
        raise ValueError(f"requested sample_id(s) not found: {sorted(missing)}")
    return selected


def _load_image_pair(draft_path: Path, target_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    with Image.open(draft_path) as draft_image, Image.open(target_path) as target_image:
        draft = np.asarray(draft_image.convert("RGB"), dtype=np.float32) / 255.0
        target = np.asarray(target_image.convert("RGB").resize(draft_image.size), dtype=np.float32) / 255.0
    if draft.shape != target.shape:
        raise ValueError("draft and target image sizes differ")
    return draft, target


def _write_target_image(sample_dir: Path, sample: dict[str, Any], output_dir: Path, target_mode: str) -> Path:
    if target_mode == TARGET_MODE_TARGET_IMAGE:
        target_name = sample.get("target_image")
        if target_name:
            target_path = sample_dir / str(target_name)
            if not target_path.exists():
                raise ValueError(f"target image does not exist: {target_path}")
            shutil.copy2(target_path, output_dir / "target.png")
            return output_dir / "target.png"
        target_mode = TARGET_MODE_SOURCE_IMAGE
    if target_mode == TARGET_MODE_FINISHED_IMAGE:
        target_path = sample_dir / sample["finished_image"]
        shutil.copy2(target_path, output_dir / "target.png")
        return output_dir / "target.png"
    if target_mode == TARGET_MODE_SOURCE_IMAGE:
        source_image = sample.get("source_image")
        if not source_image:
            raise ValueError("source-image target mode requires sample source_image metadata")
        source_path = Path(str(source_image)).expanduser()
        if not source_path.exists():
            raise ValueError(f"source image does not exist: {source_path}")
        _write_resized_source_image_target(
            source_path=source_path,
            draft_path=sample_dir / sample["draft_image"],
            output_path=output_dir / "target.png",
        )
        return output_dir / "target.png"
    raise ValueError(f"unknown target mode: {target_mode}")


def _write_resized_source_image_target(source_path: Path, draft_path: Path, output_path: Path) -> None:
    from PIL import Image

    with Image.open(draft_path) as draft_image, Image.open(source_path) as source_image:
        target = source_image.convert("RGB").resize(draft_image.size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target.save(output_path)


def _mean_absolute_difference(first_path: Path, second_path: Path) -> float:
    first, second = _load_image_pair(first_path, second_path)
    return float(np.abs(second - first).mean() * 255.0)


def _offsets(size: int, cell_size: int, stride: int) -> list[int]:
    if size <= cell_size:
        return [0]
    offsets = list(range(0, size - cell_size + 1, stride))
    last = size - cell_size
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def _weighted_patch_orientation(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> float:
    if xs.size < 2:
        return 0.0
    x = xs.astype(np.float64) + 0.5
    y = ys.astype(np.float64) + 0.5
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 0.0
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    centered_x = x - mean_x
    centered_y = y - mean_y
    cov_xx = float(np.average(centered_x * centered_x, weights=weights))
    cov_yy = float(np.average(centered_y * centered_y, weights=weights))
    cov_xy = float(np.average(centered_x * centered_y, weights=weights))
    return 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)


def _weighted_patch_extent(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    *,
    orientation_radians: float,
    min_stroke_pixels: float,
    cell_size: float,
    stroke_scale: float,
    aspect_ratio: float,
) -> tuple[float, float]:
    x = xs.astype(np.float64) + 0.5
    y = ys.astype(np.float64) + 0.5
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    cos_theta = math.cos(orientation_radians)
    sin_theta = math.sin(orientation_radians)
    centered_x = x - mean_x
    centered_y = y - mean_y
    along = centered_x * cos_theta + centered_y * sin_theta
    across = -centered_x * sin_theta + centered_y * cos_theta
    along_extent = float(along.max() - along.min() + 1.0)
    across_extent = float(across.max() - across.min() + 1.0)
    fallback_length = cell_size * stroke_scale
    fallback_width = fallback_length / aspect_ratio
    major_pixels = max(min_stroke_pixels, along_extent * stroke_scale, fallback_length)
    minor_pixels = max(min_stroke_pixels, across_extent * stroke_scale, fallback_width)
    if major_pixels < minor_pixels:
        major_pixels, minor_pixels = minor_pixels, major_pixels
    max_minor = max(min_stroke_pixels, major_pixels / aspect_ratio)
    minor_pixels = min(minor_pixels, max_minor)
    return major_pixels, minor_pixels


def _brushwright_angle(orientation_radians: float) -> float:
    return (orientation_radians / (2.0 * math.pi)) % 1.0


def _status_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        histogram[status] = histogram.get(status, 0) + 1
    return dict(sorted(histogram.items()))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _validate_config(config: ImageDeltaStrokeConfig) -> None:
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if config.stride <= 0:
        raise ValueError("stride must be positive")
    if config.max_strokes <= 0:
        raise ValueError("max_strokes must be positive")
    if not 0.0 <= config.min_error <= 1.0:
        raise ValueError("min_error must be between 0 and 1")
    if config.min_cell_changed_pixels <= 0:
        raise ValueError("min_cell_changed_pixels must be positive")
    if config.min_stroke_pixels <= 0:
        raise ValueError("min_stroke_pixels must be positive")
    if config.stroke_scale <= 0.0:
        raise ValueError("stroke_scale must be positive")
    if config.aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    if not 0.0 <= config.opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")
    if config.target_mode not in (TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE):
        raise ValueError(
            "target_mode must be "
            f"{TARGET_MODE_TARGET_IMAGE!r}, {TARGET_MODE_FINISHED_IMAGE!r}, or {TARGET_MODE_SOURCE_IMAGE!r}"
        )
    if config.recursive_passes <= 0:
        raise ValueError("recursive_passes must be positive")
    if config.min_pass_mad_improvement < 0.0:
        raise ValueError("min_pass_mad_improvement must be non-negative")
    if config.target_mad_threshold < 0.0:
        raise ValueError("target_mad_threshold must be non-negative")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export deterministic image-delta stroke predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-strokes", type=int, default=DEFAULT_MAX_STROKES)
    parser.add_argument("--min-error", type=float, default=DEFAULT_MIN_ERROR)
    parser.add_argument("--min-cell-changed-pixels", type=int, default=DEFAULT_MIN_CELL_CHANGED_PIXELS)
    parser.add_argument("--min-stroke-pixels", type=int, default=DEFAULT_MIN_STROKE_PIXELS)
    parser.add_argument("--stroke-scale", type=float, default=DEFAULT_STROKE_SCALE)
    parser.add_argument("--aspect-ratio", type=float, default=DEFAULT_ASPECT_RATIO)
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    parser.add_argument("--brush", default=DEFAULT_BRUSH)
    parser.add_argument("--require-target-contract", default=DEFAULT_TARGET_CONTRACT)
    parser.add_argument("--min-changed-pixel-ratio", type=float, default=DEFAULT_MIN_CHANGED_PIXEL_RATIO)
    parser.add_argument("--min-gradient-improvement", type=float, default=DEFAULT_MIN_GRADIENT_IMPROVEMENT)
    parser.add_argument("--min-edge-overlap", type=float, default=DEFAULT_MIN_EDGE_OVERLAP)
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE),
        default=DEFAULT_TARGET_MODE,
    )
    parser.add_argument("--recursive-passes", type=int, default=DEFAULT_RECURSIVE_PASSES)
    parser.add_argument("--stop-on-non-improvement", action=argparse.BooleanOptionalAction, default=DEFAULT_STOP_ON_NON_IMPROVEMENT)
    parser.add_argument("--min-pass-mad-improvement", type=float, default=DEFAULT_MIN_PASS_MAD_IMPROVEMENT)
    parser.add_argument("--target-mad-threshold", type=float, default=DEFAULT_TARGET_MAD_THRESHOLD)
    parser.add_argument("--no-render", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
