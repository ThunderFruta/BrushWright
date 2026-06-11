"""Export greedy biggest-improving BrushWright stroke predictions.

This is a classical renderer-scored optimizer for the visual-delta task. It
keeps BrushWright's stroke-program boundary while replacing fixed grid stamps
with iterative candidate search: propose large strokes over high-error regions,
accept the largest stroke tier that improves the target distance, then repeat.
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

from Source.Model.export_image_delta_strokes import (
    DEFAULT_TARGET_CONTRACT,
    TARGET_MODE_FINISHED_IMAGE,
    TARGET_MODE_SOURCE_IMAGE,
    _read_json,
    _select_manifest_samples,
    _write_json,
    _write_target_image,
)
from Source.Model.export_test_predictions import _program_like, _write_comparison_strip
from Source.Model.prediction_diagnostics import compute_prediction_diagnostics
from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/GreedyStrokeOptimizerV1")
DEFAULT_SPLIT = "Test"
DEFAULT_LIMIT = 4
DEFAULT_TARGET_MODE = TARGET_MODE_SOURCE_IMAGE
DEFAULT_SIZE_TIERS = (96, 72, 56, 40, 28, 18, 10, 6)
DEFAULT_DETAIL_SIZE_TIERS = (28, 18, 10, 6)
DEFAULT_ANGLE_DEGREES = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
DEFAULT_OPACITIES = (0.45, 0.65, 0.85)
DEFAULT_MAX_STROKES = 256
DEFAULT_MIN_ERROR = 0.025
DEFAULT_MIN_STROKE_MAD_IMPROVEMENT = 0.03
DEFAULT_DETAIL_MIN_STROKE_MAD_IMPROVEMENT = 0.006
DEFAULT_TARGET_MAD_THRESHOLD = 3.0
DEFAULT_DETAIL_START_STROKE = 8
DEFAULT_DETAIL_CADENCE = 2
DEFAULT_FORCE_MAX_STROKES = False
DEFAULT_MAX_COMPONENT_ANCHORS = 8
DEFAULT_MAX_POINT_ANCHORS = 16
DEFAULT_ANCHOR_GRID_SIZE = 32
DEFAULT_ANCHOR_BORDER_MARGIN = 0
DEFAULT_ASPECT_RATIO = 2.25
DEFAULT_MIN_STROKE_PIXELS = 4
DEFAULT_OUTSIDE_MASK_PENALTY = 0.25
DEFAULT_BRUSH = "paint_transformer_rect"
DEFAULT_MIN_CHANGED_PIXEL_RATIO = 0.01
DEFAULT_MIN_GRADIENT_IMPROVEMENT = 0.0
DEFAULT_MIN_EDGE_OVERLAP = 0.02
PIXEL_SCALE = 255.0


@dataclass(frozen=True)
class GreedyStrokeOptimizerConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    limit: int = DEFAULT_LIMIT
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
    min_changed_pixel_ratio: float = DEFAULT_MIN_CHANGED_PIXEL_RATIO
    min_gradient_improvement: float = DEFAULT_MIN_GRADIENT_IMPROVEMENT
    min_edge_overlap: float = DEFAULT_MIN_EDGE_OVERLAP
    render: bool = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exported = export_greedy_stroke_optimizer(
        GreedyStrokeOptimizerConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            split=args.split,
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
            min_changed_pixel_ratio=args.min_changed_pixel_ratio,
            min_gradient_improvement=args.min_gradient_improvement,
            min_edge_overlap=args.min_edge_overlap,
            render=not args.no_render,
        )
    )
    print(json.dumps({"exported": exported}, indent=2), flush=True)
    return 0


def export_greedy_stroke_optimizer(config: GreedyStrokeOptimizerConfig) -> list[dict[str, Any]]:
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
        print(f"[{index}/{len(samples)}] optimizing {sample['sample_id']} -> {output_dir}", flush=True)
        exported.append(_export_sample(sample_dir, sample, output_dir, config))

    _write_json(
        prepared_root / "export_manifest.json",
        {"version": 1, "split": config.split, "summary": _export_summary(exported), "samples": exported},
    )
    return exported


def optimize_greedy_strokes(
    draft_path: Path,
    target_path: Path,
    *,
    size_tiers: tuple[int, ...] = DEFAULT_SIZE_TIERS,
    detail_size_tiers: tuple[int, ...] = DEFAULT_DETAIL_SIZE_TIERS,
    angle_degrees: tuple[float, ...] = DEFAULT_ANGLE_DEGREES,
    opacities: tuple[float, ...] = DEFAULT_OPACITIES,
    max_strokes: int = DEFAULT_MAX_STROKES,
    min_error: float = DEFAULT_MIN_ERROR,
    min_stroke_mad_improvement: float = DEFAULT_MIN_STROKE_MAD_IMPROVEMENT,
    detail_min_stroke_mad_improvement: float = DEFAULT_DETAIL_MIN_STROKE_MAD_IMPROVEMENT,
    target_mad_threshold: float = DEFAULT_TARGET_MAD_THRESHOLD,
    detail_start_stroke: int = DEFAULT_DETAIL_START_STROKE,
    detail_cadence: int = DEFAULT_DETAIL_CADENCE,
    force_max_strokes: bool = DEFAULT_FORCE_MAX_STROKES,
    max_component_anchors: int = DEFAULT_MAX_COMPONENT_ANCHORS,
    max_point_anchors: int = DEFAULT_MAX_POINT_ANCHORS,
    anchor_grid_size: int = DEFAULT_ANCHOR_GRID_SIZE,
    anchor_border_margin: int = DEFAULT_ANCHOR_BORDER_MARGIN,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    min_stroke_pixels: int = DEFAULT_MIN_STROKE_PIXELS,
    outside_mask_penalty: float = DEFAULT_OUTSIDE_MASK_PENALTY,
    brush: str = DEFAULT_BRUSH,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    current, target = _load_image_pair(draft_path, target_path)
    height, width, _ = current.shape
    denominator = float(height * width * 3)
    current_error_sum = float(np.abs(target - current).sum())
    current_mad = current_error_sum / denominator * PIXEL_SCALE
    base_error = np.max(np.abs(target - current), axis=2)
    edit_mask = base_error > min_error
    initial_mad = current_mad
    accepted: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stop_reason = "max_strokes"
    total_candidates_scored = 0

    for stroke_index in range(max_strokes):
        if current_mad <= target_mad_threshold:
            stop_reason = "target_threshold"
            break

        error = np.max(np.abs(target - current), axis=2)
        active_mask = error > min_error
        anchor_mask = _apply_anchor_border_margin(active_mask, anchor_border_margin)
        if not bool(anchor_mask.any()):
            anchor_mask = active_mask
        anchors = _greedy_anchors(
            error,
            anchor_mask,
            max_component_anchors=max_component_anchors,
            max_point_anchors=max_point_anchors,
            anchor_grid_size=anchor_grid_size,
        )
        if not anchors:
            stop_reason = "no_error_anchors"
            break
        detail_priority = _detail_priority(current, target, error)
        detail_anchors = _greedy_anchors(
            detail_priority,
            anchor_mask,
            max_component_anchors=max_component_anchors,
            max_point_anchors=max_point_anchors,
            anchor_grid_size=anchor_grid_size,
            profile="detail",
        )
        if not detail_anchors:
            detail_anchors = anchors

        selected: dict[str, Any] | None = None
        fallback_selected: dict[str, Any] | None = None
        tier_entries: list[dict[str, Any]] = []
        for tier_group in _tier_search_groups(
            stroke_index=stroke_index + 1,
            size_tiers=size_tiers,
            detail_size_tiers=detail_size_tiers,
            detail_start_stroke=detail_start_stroke,
            detail_cadence=detail_cadence,
            coarse_min_improvement=min_stroke_mad_improvement,
            detail_min_improvement=detail_min_stroke_mad_improvement,
        ):
            group_anchors = detail_anchors if tier_group["phase"] == "detail" else anchors
            for size_pixels in tier_group["size_tiers"]:
                best_for_tier: dict[str, Any] | None = None
                tier_candidate_count = 0
                for anchor in group_anchors:
                    for angle_degrees_value in angle_degrees:
                        angle_turns = (angle_degrees_value / 360.0) % 1.0
                        for opacity in opacities:
                            candidate = _score_candidate(
                                current=current,
                                target=target,
                                edit_mask=edit_mask,
                                current_error_sum=current_error_sum,
                                denominator=denominator,
                                anchor=anchor,
                                size_pixels=float(size_pixels),
                                angle_turns=angle_turns,
                                opacity=float(opacity),
                                aspect_ratio=aspect_ratio,
                                min_stroke_pixels=float(min_stroke_pixels),
                                outside_mask_penalty=outside_mask_penalty,
                                brush=brush,
                            )
                            tier_candidate_count += 1
                            if candidate is None:
                                continue
                            if best_for_tier is None or float(candidate["score"]) > float(best_for_tier["score"]):
                                best_for_tier = candidate
                total_candidates_scored += tier_candidate_count
                tier_entries.append(
                    {
                        "phase": tier_group["phase"],
                        "anchor_profile": tier_group["anchor_profile"],
                        "size_pixels": int(size_pixels),
                        "candidate_count": tier_candidate_count,
                        "min_improvement": tier_group["min_improvement"],
                        "best_improvement": None if best_for_tier is None else best_for_tier["improvement"],
                        "best_score": None if best_for_tier is None else best_for_tier["score"],
                    }
                )
                if best_for_tier is not None and (
                    fallback_selected is None or float(best_for_tier["score"]) > float(fallback_selected["score"])
                ):
                    fallback_selected = best_for_tier
                    fallback_selected["phase"] = tier_group["phase"]
                if best_for_tier is not None and float(best_for_tier["improvement"]) >= float(tier_group["min_improvement"]):
                    selected = best_for_tier
                    selected["phase"] = tier_group["phase"]
                    break
            if selected is not None:
                break
        forced_accept = False
        if selected is None and force_max_strokes and fallback_selected is not None:
            selected = fallback_selected
            forced_accept = True

        if selected is None:
            stop_reason = "no_improving_candidate"
            history.append(
                {
                    "stroke_index": stroke_index + 1,
                    "accepted": False,
                    "previous_mad": current_mad,
                    "tier_search": tier_entries,
                }
            )
            break

        y_slice, x_slice = selected["bbox_slices"]
        current[y_slice, x_slice] = selected["rendered_patch"]
        current_error_sum = float(selected["candidate_error_sum"])
        previous_mad = current_mad
        current_mad = float(selected["candidate_mad"])
        accepted.append(dict(selected["stroke"]))
        history.append(
            {
                "stroke_index": stroke_index + 1,
                "accepted": True,
                "previous_mad": previous_mad,
                "current_mad": current_mad,
                "improvement": float(selected["improvement"]),
                "score": float(selected["score"]),
                "forced_accept": forced_accept,
                "size_pixels": int(selected["size_pixels"]),
                "phase": selected["phase"],
                "anchor_profile": selected["anchor"]["profile"],
                "opacity": float(selected["stroke"]["opacity"]),
                "anchor_source": selected["anchor"]["source"],
                "anchor_score": float(selected["anchor"]["score"]),
                "tier_search": tier_entries,
            }
        )

    final_image = current
    manifest = {
        "version": 1,
        "initial_mad": initial_mad,
        "final_mad_estimate": current_mad,
        "estimated_mad_improvement": initial_mad - current_mad,
        "accepted_stroke_count": len(accepted),
        "max_strokes": max_strokes,
        "stop_reason": stop_reason,
        "total_candidates_scored": total_candidates_scored,
        "size_tiers": list(size_tiers),
        "detail_size_tiers": list(detail_size_tiers),
        "angle_degrees": list(angle_degrees),
        "opacities": list(opacities),
        "min_stroke_mad_improvement": min_stroke_mad_improvement,
        "detail_min_stroke_mad_improvement": detail_min_stroke_mad_improvement,
        "detail_start_stroke": detail_start_stroke,
        "detail_cadence": detail_cadence,
        "force_max_strokes": force_max_strokes,
        "anchor_border_margin": anchor_border_margin,
        "target_mad_threshold": target_mad_threshold,
        "history": history,
    }
    return accepted, manifest, final_image


def propose_greedy_candidates(
    draft_path: Path,
    target_path: Path,
    *,
    size_tiers: tuple[int, ...] = DEFAULT_SIZE_TIERS,
    max_component_anchors: int = DEFAULT_MAX_COMPONENT_ANCHORS,
    max_point_anchors: int = DEFAULT_MAX_POINT_ANCHORS,
    anchor_grid_size: int = DEFAULT_ANCHOR_GRID_SIZE,
    min_error: float = DEFAULT_MIN_ERROR,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    min_stroke_pixels: int = DEFAULT_MIN_STROKE_PIXELS,
    brush: str = DEFAULT_BRUSH,
) -> list[dict[str, Any]]:
    draft, target = _load_image_pair(draft_path, target_path)
    height, width, _ = draft.shape
    error = np.max(np.abs(target - draft), axis=2)
    anchors = _greedy_anchors(
        error,
        error > min_error,
        max_component_anchors=max_component_anchors,
        max_point_anchors=max_point_anchors,
        anchor_grid_size=anchor_grid_size,
    )
    candidates = []
    for anchor in anchors:
        for size_pixels in size_tiers:
            length_pixels = float(size_pixels)
            width_pixels = max(float(min_stroke_pixels), length_pixels / aspect_ratio)
            candidates.append(
                {
                    "stroke": {
                        "x": _clamp(anchor["x"] / width),
                        "y": _clamp(anchor["y"] / height),
                        "angle": 0.0,
                        "length": _clamp(length_pixels / width),
                        "width": _clamp(width_pixels / height),
                        "color": [0.0, 0.0, 0.0],
                        "opacity": DEFAULT_OPACITIES[0],
                        "brush": brush,
                    },
                    "size_pixels": int(size_pixels),
                    "anchor": anchor,
                }
            )
    return candidates


def _export_sample(
    sample_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    config: GreedyStrokeOptimizerConfig,
) -> dict[str, Any]:
    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    target_path = _write_target_image(sample_dir, sample, output_dir, config.target_mode)

    predicted_strokes, greedy_manifest, _ = optimize_greedy_strokes(
        output_dir / "draft.png",
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
    metadata = {
        **dict(target_program.get("metadata", {})),
        "prediction_source": "greedy_stroke_optimizer_v1",
        "sample_id": sample["sample_id"],
        "split": "greedy_optimizer_added_strokes",
        "target_mode": config.target_mode,
        "optimizer": {
            "max_strokes": config.max_strokes,
            "size_tiers": list(config.size_tiers),
            "detail_size_tiers": list(config.detail_size_tiers),
            "angle_degrees": list(config.angle_degrees),
            "opacities": list(config.opacities),
            "min_stroke_mad_improvement": config.min_stroke_mad_improvement,
            "detail_min_stroke_mad_improvement": config.detail_min_stroke_mad_improvement,
            "target_mad_threshold": config.target_mad_threshold,
            "detail_start_stroke": config.detail_start_stroke,
            "detail_cadence": config.detail_cadence,
            "force_max_strokes": config.force_max_strokes,
            "anchor_border_margin": config.anchor_border_margin,
            "selected_count": len(predicted_strokes),
        },
    }
    predicted_finishing_program = _program_like(target_program, metadata, predicted_strokes)
    predicted_full_program = _program_like(
        base_program,
        {
            **dict(base_program.get("metadata", {})),
            "prediction_source": "greedy_stroke_optimizer_v1",
            "sample_id": sample["sample_id"],
            "split": "base_plus_greedy_optimizer_strokes",
            "target_mode": config.target_mode,
        },
        [*base_program["strokes"], *predicted_strokes],
    )
    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(
        output_dir / "sample.json",
        sample | {"prediction_source": "greedy_stroke_optimizer_v1", "prediction_target_mode": config.target_mode},
    )
    _write_json(output_dir / "greedy_manifest.json", greedy_manifest)

    if config.render:
        if predicted_strokes:
            render_program_final_with_paint_transformer(
                output_dir / "added_strokes.json",
                output_dir / "predicted.png",
                background_path=output_dir / "draft.png",
            )
        else:
            shutil.copy2(output_dir / "draft.png", output_dir / "predicted.png")
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
        diagnostics["greedy_optimizer"] = {
            "estimated_initial_mad": greedy_manifest["initial_mad"],
            "estimated_final_mad": greedy_manifest["final_mad_estimate"],
            "estimated_mad_improvement": greedy_manifest["estimated_mad_improvement"],
            "stop_reason": greedy_manifest["stop_reason"],
            "total_candidates_scored": greedy_manifest["total_candidates_scored"],
        }
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
        "greedy_manifest": str(output_dir / "greedy_manifest.json"),
        "status": diagnostics["status"],
        "visual_improved": diagnostics["visual_improved"],
        "added_strokes": str(output_dir / "added_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
        "added_stroke_count": len(predicted_strokes),
    }


def _score_candidate(
    *,
    current: np.ndarray,
    target: np.ndarray,
    edit_mask: np.ndarray,
    current_error_sum: float,
    denominator: float,
    anchor: dict[str, Any],
    size_pixels: float,
    angle_turns: float,
    opacity: float,
    aspect_ratio: float,
    min_stroke_pixels: float,
    outside_mask_penalty: float,
    brush: str,
) -> dict[str, Any] | None:
    height, width, _ = current.shape
    length_pixels = max(min_stroke_pixels, size_pixels)
    width_pixels = max(min_stroke_pixels, length_pixels / aspect_ratio)
    bbox = _stroke_bbox(
        height=height,
        width=width,
        center_x=float(anchor["x"]),
        center_y=float(anchor["y"]),
        length_pixels=length_pixels,
        width_pixels=width_pixels,
    )
    if bbox is None:
        return None
    y_slice, x_slice = bbox
    alpha = _stroke_alpha_mask(
        height=y_slice.stop - y_slice.start,
        width=x_slice.stop - x_slice.start,
        center_x=float(anchor["x"]) - x_slice.start,
        center_y=float(anchor["y"]) - y_slice.start,
        length_pixels=length_pixels,
        width_pixels=width_pixels,
        angle_turns=angle_turns,
    )
    effective_alpha = (alpha * opacity).clip(0.0, 1.0)
    if float(effective_alpha.sum()) <= 0.0:
        return None

    current_patch = current[y_slice, x_slice]
    target_patch = target[y_slice, x_slice]
    color = _estimate_residual_color(current_patch, target_patch, effective_alpha)
    rendered_patch = _composite_color(current_patch, color, effective_alpha)
    previous_patch_error = float(np.abs(target_patch - current_patch).sum())
    candidate_patch_error = float(np.abs(target_patch - rendered_patch).sum())
    candidate_error_sum = current_error_sum - previous_patch_error + candidate_patch_error
    previous_mad = current_error_sum / denominator * PIXEL_SCALE
    candidate_mad = candidate_error_sum / denominator * PIXEL_SCALE
    improvement = previous_mad - candidate_mad
    outside_mask_patch = ~edit_mask[y_slice, x_slice]
    outside_change = 0.0
    if bool(outside_mask_patch.any()):
        outside_change = float(np.abs(rendered_patch - current_patch)[outside_mask_patch].mean() * PIXEL_SCALE)
    score = improvement - outside_mask_penalty * outside_change
    return {
        "stroke": {
            "x": _clamp(float(anchor["x"]) / width),
            "y": _clamp(float(anchor["y"]) / height),
            "angle": _clamp(angle_turns),
            "length": _clamp(length_pixels / width),
            "width": _clamp(width_pixels / height),
            "color": [_clamp(float(channel)) for channel in color.tolist()],
            "opacity": _clamp(opacity),
            "brush": brush,
        },
        "anchor": anchor,
        "bbox_slices": (y_slice, x_slice),
        "rendered_patch": rendered_patch,
        "candidate_error_sum": candidate_error_sum,
        "candidate_mad": candidate_mad,
        "improvement": improvement,
        "outside_change": outside_change,
        "score": score,
        "size_pixels": int(round(size_pixels)),
    }


def _tier_search_groups(
    *,
    stroke_index: int,
    size_tiers: tuple[int, ...],
    detail_size_tiers: tuple[int, ...],
    detail_start_stroke: int,
    detail_cadence: int,
    coarse_min_improvement: float,
    detail_min_improvement: float,
) -> list[dict[str, Any]]:
    detail_due = (
        detail_cadence > 0
        and detail_start_stroke > 0
        and stroke_index >= detail_start_stroke
        and (stroke_index - detail_start_stroke) % detail_cadence == 0
    )
    detail_group = {
        "phase": "detail",
        "anchor_profile": "detail",
        "size_tiers": tuple(size for size in detail_size_tiers if size in size_tiers),
        "min_improvement": detail_min_improvement,
    }
    coarse_group = {
        "phase": "coarse",
        "anchor_profile": "coarse",
        "size_tiers": size_tiers,
        "min_improvement": coarse_min_improvement,
    }
    if detail_due and detail_group["size_tiers"]:
        return [detail_group, coarse_group]
    return [coarse_group]


def _greedy_anchors(
    error: np.ndarray,
    mask: np.ndarray,
    *,
    max_component_anchors: int,
    max_point_anchors: int,
    anchor_grid_size: int,
    profile: str = "coarse",
) -> list[dict[str, Any]]:
    anchors = [
        *_component_anchors(error, mask, max_anchors=max_component_anchors, profile=profile),
        *_top_error_anchors(error, mask, max_anchors=max_point_anchors, grid_size=anchor_grid_size, profile=profile),
    ]
    deduped: dict[tuple[int, int], dict[str, Any]] = {}
    for anchor in anchors:
        key = (int(round(float(anchor["x"]))), int(round(float(anchor["y"]))))
        existing = deduped.get(key)
        if existing is None or float(anchor["score"]) > float(existing["score"]):
            deduped[key] = anchor
    return sorted(deduped.values(), key=lambda item: float(item["score"]), reverse=True)


def _apply_anchor_border_margin(mask: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return mask
    height, width = mask.shape
    if margin * 2 >= height or margin * 2 >= width:
        return mask
    trimmed = mask.copy()
    trimmed[:margin, :] = False
    trimmed[-margin:, :] = False
    trimmed[:, :margin] = False
    trimmed[:, -margin:] = False
    return trimmed


def _component_anchors(error: np.ndarray, mask: np.ndarray, *, max_anchors: int, profile: str = "coarse") -> list[dict[str, Any]]:
    if max_anchors <= 0 or not bool(mask.any()):
        return []
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    anchors = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        xs = []
        ys = []
        weights = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            weights.append(float(error[y, x]))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if not weights:
            continue
        weight_array = np.asarray(weights, dtype=np.float64)
        weight_sum = float(weight_array.sum())
        if weight_sum <= 0.0:
            continue
        anchors.append(
            {
                "x": float(np.average(np.asarray(xs, dtype=np.float64) + 0.5, weights=weight_array)),
                "y": float(np.average(np.asarray(ys, dtype=np.float64) + 0.5, weights=weight_array)),
                "score": weight_sum,
                "profile": profile,
                "source": "component",
                "pixel_count": len(weights),
            }
        )
    anchors.sort(key=lambda item: float(item["score"]), reverse=True)
    return anchors[:max_anchors]


def _top_error_anchors(
    error: np.ndarray,
    mask: np.ndarray,
    *,
    max_anchors: int,
    grid_size: int,
    profile: str = "coarse",
) -> list[dict[str, Any]]:
    if max_anchors <= 0 or grid_size <= 0 or not bool(mask.any()):
        return []
    height, width = mask.shape
    anchors = []
    for top in range(0, height, grid_size):
        for left in range(0, width, grid_size):
            bottom = min(height, top + grid_size)
            right = min(width, left + grid_size)
            block_mask = mask[top:bottom, left:right]
            if not bool(block_mask.any()):
                continue
            block_error = np.where(block_mask, error[top:bottom, left:right], -1.0)
            flat_index = int(np.argmax(block_error))
            y, x = np.unravel_index(flat_index, block_error.shape)
            score = float(block_error[y, x])
            anchors.append(
                {
                    "x": float(left + x + 0.5),
                    "y": float(top + y + 0.5),
                    "score": score,
                    "profile": profile,
                    "source": "top_error",
                    "pixel_count": int(block_mask.sum()),
                }
            )
    anchors.sort(key=lambda item: float(item["score"]), reverse=True)
    return anchors[:max_anchors]


def _detail_priority(current: np.ndarray, target: np.ndarray, error: np.ndarray) -> np.ndarray:
    target_edges = _gradient_magnitude(target)
    residual_edges = _gradient_magnitude(np.abs(target - current))
    edge_priority = 0.65 * target_edges + 0.35 * residual_edges
    edge_max = float(edge_priority.max())
    if edge_max > 0.0:
        edge_priority = edge_priority / edge_max
    return error * (0.20 + edge_priority)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2) if image.ndim == 3 else image
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _stroke_bbox(
    *,
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    length_pixels: float,
    width_pixels: float,
) -> tuple[slice, slice] | None:
    radius = 0.5 * math.hypot(length_pixels, width_pixels) + 2.0
    left = max(0, int(math.floor(center_x - radius)))
    right = min(width, int(math.ceil(center_x + radius)))
    top = max(0, int(math.floor(center_y - radius)))
    bottom = min(height, int(math.ceil(center_y + radius)))
    if left >= right or top >= bottom:
        return None
    return slice(top, bottom), slice(left, right)


def _stroke_alpha_mask(
    *,
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    length_pixels: float,
    width_pixels: float,
    angle_turns: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx + 0.5 - center_x
    y = yy + 0.5 - center_y
    radians = angle_turns * 2.0 * math.pi
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    along = x * cos_theta + y * sin_theta
    across = -x * sin_theta + y * cos_theta
    half_length = max(1.0, length_pixels * 0.5)
    half_width = max(1.0, width_pixels * 0.5)
    normalized = (along / half_length) ** 2 + (across / half_width) ** 2
    return np.clip((1.08 - normalized) / 0.18, 0.0, 1.0).astype(np.float32)


def _estimate_residual_color(current_patch: np.ndarray, target_patch: np.ndarray, effective_alpha: np.ndarray) -> np.ndarray:
    alpha = effective_alpha[..., None].astype(np.float32)
    denominator = float((alpha * alpha).sum())
    if denominator <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    numerator = (alpha * (target_patch - current_patch * (1.0 - alpha))).sum(axis=(0, 1))
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def _composite_color(current_patch: np.ndarray, color: np.ndarray, effective_alpha: np.ndarray) -> np.ndarray:
    alpha = effective_alpha[..., None].astype(np.float32)
    return np.clip(color.reshape(1, 1, 3) * alpha + current_patch * (1.0 - alpha), 0.0, 1.0)


def _load_image_pair(first_path: Path, second_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for greedy stroke optimization") from exc

    with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
        first = np.asarray(first_image.convert("RGB"), dtype=np.float32) / PIXEL_SCALE
        second = np.asarray(second_image.convert("RGB").resize(first_image.size), dtype=np.float32) / PIXEL_SCALE
    return first, second


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
        "checkpoint_status": "visual_passed" if visual_pass else "visual_failed",
    }


def _validate_config(config: GreedyStrokeOptimizerConfig) -> None:
    if config.limit <= 0 and not config.sample_ids:
        raise ValueError("limit must be positive unless sample_ids are provided")
    if config.target_mode not in (TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE):
        raise ValueError(f"target_mode must be one of {TARGET_MODE_FINISHED_IMAGE!r}, {TARGET_MODE_SOURCE_IMAGE!r}")
    if not config.size_tiers or any(size <= 0 for size in config.size_tiers):
        raise ValueError("size_tiers must contain positive values")
    if tuple(config.size_tiers) != tuple(sorted(config.size_tiers, reverse=True)):
        raise ValueError("size_tiers must be sorted from largest to smallest")
    if not config.detail_size_tiers or any(size <= 0 for size in config.detail_size_tiers):
        raise ValueError("detail_size_tiers must contain positive values")
    if tuple(config.detail_size_tiers) != tuple(sorted(config.detail_size_tiers, reverse=True)):
        raise ValueError("detail_size_tiers must be sorted from largest to smallest")
    if any(size not in config.size_tiers for size in config.detail_size_tiers):
        raise ValueError("detail_size_tiers must be a subset of size_tiers")
    if not config.angle_degrees:
        raise ValueError("angle_degrees must not be empty")
    if not config.opacities or any(opacity <= 0.0 or opacity > 1.0 for opacity in config.opacities):
        raise ValueError("opacities must be in the range (0, 1]")
    if config.max_strokes <= 0:
        raise ValueError("max_strokes must be positive")
    if config.min_error < 0.0:
        raise ValueError("min_error must be non-negative")
    if config.min_stroke_mad_improvement < 0.0:
        raise ValueError("min_stroke_mad_improvement must be non-negative")
    if config.detail_min_stroke_mad_improvement < 0.0:
        raise ValueError("detail_min_stroke_mad_improvement must be non-negative")
    if config.target_mad_threshold < 0.0:
        raise ValueError("target_mad_threshold must be non-negative")
    if config.detail_start_stroke < 1:
        raise ValueError("detail_start_stroke must be at least 1")
    if config.detail_cadence < 0:
        raise ValueError("detail_cadence must be non-negative")
    if config.max_component_anchors < 0 or config.max_point_anchors < 0:
        raise ValueError("anchor counts must be non-negative")
    if config.max_component_anchors == 0 and config.max_point_anchors == 0:
        raise ValueError("at least one anchor source must be enabled")
    if config.anchor_grid_size <= 0:
        raise ValueError("anchor_grid_size must be positive")
    if config.anchor_border_margin < 0:
        raise ValueError("anchor_border_margin must be non-negative")
    if config.aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    if config.min_stroke_pixels <= 0:
        raise ValueError("min_stroke_pixels must be positive")
    if config.outside_mask_penalty < 0.0:
        raise ValueError("outside_mask_penalty must be non-negative")


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export greedy biggest-improving stroke optimizer predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--target-mode", choices=(TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE), default=DEFAULT_TARGET_MODE)
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
    parser.add_argument("--min-changed-pixel-ratio", type=float, default=DEFAULT_MIN_CHANGED_PIXEL_RATIO)
    parser.add_argument("--min-gradient-improvement", type=float, default=DEFAULT_MIN_GRADIENT_IMPROVEMENT)
    parser.add_argument("--min-edge-overlap", type=float, default=DEFAULT_MIN_EDGE_OVERLAP)
    parser.add_argument("--no-render", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
