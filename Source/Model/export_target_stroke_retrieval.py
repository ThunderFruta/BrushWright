"""Export target-stroke retrieval baselines for BrushWright V6.

This is an oracle-style compiler: it does not invent strokes. It selects real
withheld PaintTransformer strokes from finishing_strokes.json, ranks them by
the current draft-to-target error, and renders selected strokes recursively.
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
from Source.Model.export_image_delta_strokes import (
    TARGET_MODE_FINISHED_IMAGE,
    TARGET_MODE_TARGET_IMAGE,
    TARGET_MODE_SOURCE_IMAGE,
    _mean_absolute_difference,
    _select_manifest_samples,
    _write_target_image,
)
from Source.Model.prediction_diagnostics import compute_prediction_diagnostics
from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/TargetStrokeRetrievalV6Oracle")
DEFAULT_SPLIT = "Test"
DEFAULT_LIMIT = 4
DEFAULT_RECURSIVE_PASSES = 6
DEFAULT_STROKES_PER_PASS = 256
DEFAULT_MIN_CHANGED_PIXEL_RATIO = 0.01
DEFAULT_MIN_GRADIENT_IMPROVEMENT = 0.0
DEFAULT_MIN_EDGE_OVERLAP = 0.02
DEFAULT_TARGET_CONTRACT = "paint_transformer_original_image_target_v1"
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MIN_PASS_MAD_IMPROVEMENT = 0.0
STROKE_SOURCE_FINISHING = "finishing"
STROKE_SOURCE_FULL_PROGRAM = "full-program"
DEFAULT_STROKE_SOURCE = STROKE_SOURCE_FINISHING
DEFAULT_TARGET_MODE = TARGET_MODE_TARGET_IMAGE


@dataclass(frozen=True)
class TargetStrokeRetrievalConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    limit: int = DEFAULT_LIMIT
    sample_ids: tuple[str, ...] = ()
    recursive_passes: int = DEFAULT_RECURSIVE_PASSES
    strokes_per_pass: int = DEFAULT_STROKES_PER_PASS
    min_score: float = DEFAULT_MIN_SCORE
    min_pass_mad_improvement: float = DEFAULT_MIN_PASS_MAD_IMPROVEMENT
    stroke_source: str = DEFAULT_STROKE_SOURCE
    target_mode: str = DEFAULT_TARGET_MODE
    stop_on_no_candidates: bool = True
    require_target_contract: str | None = DEFAULT_TARGET_CONTRACT
    min_changed_pixel_ratio: float = DEFAULT_MIN_CHANGED_PIXEL_RATIO
    min_gradient_improvement: float = DEFAULT_MIN_GRADIENT_IMPROVEMENT
    min_edge_overlap: float = DEFAULT_MIN_EDGE_OVERLAP
    render: bool = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exported = export_target_stroke_retrieval(
        TargetStrokeRetrievalConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            split=args.split,
            limit=args.limit,
            sample_ids=tuple(args.sample_id or ()),
            recursive_passes=args.recursive_passes,
            strokes_per_pass=args.strokes_per_pass,
            min_score=args.min_score,
            min_pass_mad_improvement=args.min_pass_mad_improvement,
            stroke_source=args.stroke_source,
            target_mode=args.target_mode,
            stop_on_no_candidates=not args.keep_empty_passes,
            require_target_contract=args.require_target_contract,
            min_changed_pixel_ratio=args.min_changed_pixel_ratio,
            min_gradient_improvement=args.min_gradient_improvement,
            min_edge_overlap=args.min_edge_overlap,
            render=not args.no_render,
        )
    )
    print(json.dumps({"exported": exported}, indent=2), flush=True)
    return 0


def export_target_stroke_retrieval(config: TargetStrokeRetrievalConfig) -> list[dict[str, Any]]:
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
        print(f"[{index}/{len(samples)}] retrieving {sample['sample_id']} -> {output_dir}", flush=True)
        exported.append(_export_sample(sample_dir, sample, output_dir, config))

    _write_json(
        prepared_root / "export_manifest.json",
        {"version": 1, "split": config.split, "summary": _export_summary(exported), "samples": exported},
    )
    return exported


def _export_sample(
    sample_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    config: TargetStrokeRetrievalConfig,
) -> dict[str, Any]:
    if not config.render:
        raise ValueError("target-stroke retrieval export requires rendering")
    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    candidate_program = _candidate_program(sample_dir, sample, config.stroke_source)
    remaining = list(candidate_program["strokes"])
    selected_strokes: list[dict[str, Any]] = []
    pass_entries: list[dict[str, Any]] = []
    stop_reason = "max_passes"

    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    _write_target_image(sample_dir, sample, output_dir, config.target_mode)
    current_draft_path = output_dir / "draft.png"
    previous_mad = _mean_absolute_difference(current_draft_path, output_dir / "target.png")

    for pass_index in range(1, config.recursive_passes + 1):
        pass_dir = output_dir / f"pass_{pass_index:04d}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_draft_path, pass_dir / "draft.png")
        shutil.copy2(output_dir / "target.png", pass_dir / "target.png")

        ranked = rank_target_strokes(
            strokes=remaining,
            draft_path=current_draft_path,
            target_path=output_dir / "target.png",
            min_score=config.min_score,
        )
        pass_strokes = [entry["stroke"] for entry in ranked[: config.strokes_per_pass]]
        selected_indices = {int(entry["source_index"]) for entry in ranked[: config.strokes_per_pass]}

        pass_program = _program_like(
            template=target_program,
            metadata={
                **dict(target_program.get("metadata", {})),
                "prediction_source": "target_stroke_retrieval_v6_oracle",
                "sample_id": sample["sample_id"],
                "split": "target_stroke_retrieval_pass",
                "recursive_pass": pass_index,
                "stroke_source": config.stroke_source,
                "target_mode": config.target_mode,
                "candidate_count": len(ranked),
                "selected_count": len(pass_strokes),
                "min_score": config.min_score,
            },
            strokes=pass_strokes,
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
        _write_comparison_strip(
            pass_dir / "draft.png",
            pass_dir / "target.png",
            pass_dir / "predicted.png",
            pass_dir / "comparison.png",
        )
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
        accepted = bool(pass_strokes) and pass_improvement >= config.min_pass_mad_improvement
        diagnostics["target_stroke_retrieval"] = {
            "pass_index": pass_index,
            "candidate_count": len(ranked),
            "selected_count": len(pass_strokes),
            "remaining_count": len(remaining),
            "max_score": float(ranked[0]["score"]) if ranked else 0.0,
            "min_selected_score": float(ranked[min(len(pass_strokes), len(ranked)) - 1]["score"]) if pass_strokes else 0.0,
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
            stop_reason = "non_improvement" if pass_strokes else "no_candidates"
            break
        selected_strokes.extend(pass_strokes)
        remaining = [stroke for index, stroke in enumerate(remaining) if index not in selected_indices]
        current_draft_path = pass_dir / "predicted.png"
        previous_mad = current_mad
        if not pass_strokes and config.stop_on_no_candidates:
            stop_reason = "no_candidates"
            break
        if not remaining:
            stop_reason = "no_remaining_strokes"
            break

    predicted_finishing_program = _program_like(
        template=target_program,
        metadata={
            **dict(target_program.get("metadata", {})),
            "prediction_source": "target_stroke_retrieval_v6_oracle",
            "sample_id": sample["sample_id"],
            "split": "target_stroke_retrieval_added_strokes",
            "stroke_source": config.stroke_source,
            "target_mode": config.target_mode,
            "recursive_passes_requested": config.recursive_passes,
            "recursive_passes_completed": len(pass_entries),
            "strokes_per_pass": config.strokes_per_pass,
            "stop_reason": stop_reason,
            "min_pass_mad_improvement": config.min_pass_mad_improvement,
        },
        strokes=selected_strokes,
    )
    predicted_full_program = _program_like(
        template=base_program,
        metadata={
            **dict(base_program.get("metadata", {})),
            "prediction_source": "target_stroke_retrieval_v6_oracle",
            "sample_id": sample["sample_id"],
            "split": "base_plus_target_stroke_retrieval",
            "stroke_source": config.stroke_source,
            "target_mode": config.target_mode,
        },
        strokes=[*base_program["strokes"], *selected_strokes],
    )
    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(
        output_dir / "sample.json",
        sample
        | {
            "prediction_source": "target_stroke_retrieval_v6_oracle",
            "prediction_stroke_source": config.stroke_source,
            "prediction_target_mode": config.target_mode,
        },
    )
    shutil.copy2(current_draft_path, output_dir / "predicted.png")
    _write_comparison_strip(output_dir / "draft.png", output_dir / "target.png", output_dir / "predicted.png", output_dir / "comparison.png")
    diagnostics = compute_prediction_diagnostics(
        draft_path=output_dir / "draft.png",
        target_path=output_dir / "target.png",
        predicted_path=output_dir / "predicted.png",
        predicted_strokes=selected_strokes,
        target_strokes=target_program["strokes"] if config.target_mode == TARGET_MODE_FINISHED_IMAGE else None,
        min_changed_pixel_ratio=config.min_changed_pixel_ratio,
        min_gradient_improvement=config.min_gradient_improvement,
        min_edge_overlap=config.min_edge_overlap,
    )
    diagnostics["target_stroke_retrieval"] = {
        "passes": pass_entries,
        "stop_reason": stop_reason,
        "total_added_strokes": len(selected_strokes),
        "remaining_count": len(remaining),
        "stroke_source": config.stroke_source,
        "target_mode": config.target_mode,
        "min_pass_mad_improvement": config.min_pass_mad_improvement,
    }
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(
        output_dir / "retrieval_manifest.json",
        {
            "version": 1,
            "sample_id": sample["sample_id"],
            "source": "target_stroke_retrieval_v6_oracle",
            "stroke_source": config.stroke_source,
            "target_mode": config.target_mode,
            "min_pass_mad_improvement": config.min_pass_mad_improvement,
            "stop_reason": stop_reason,
            "total_added_strokes": len(selected_strokes),
            "remaining_count": len(remaining),
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
        "status": diagnostics["status"],
        "visual_improved": diagnostics["visual_improved"],
        "added_strokes": str(output_dir / "added_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
        "retrieval_manifest": str(output_dir / "retrieval_manifest.json"),
        "recursive_passes_completed": len(pass_entries),
        "retrieval_stop_reason": stop_reason,
        "added_stroke_count": len(selected_strokes),
    }


def _candidate_program(sample_dir: Path, sample: dict[str, Any], stroke_source: str) -> dict[str, Any]:
    if stroke_source == STROKE_SOURCE_FINISHING:
        return _read_json(sample_dir / sample["finishing_strokes"])
    if stroke_source == STROKE_SOURCE_FULL_PROGRAM:
        return _read_json(sample_dir / sample["full_program"])
    raise ValueError(f"unknown stroke source: {stroke_source}")


def rank_target_strokes(
    strokes: Sequence[dict[str, Any]],
    draft_path: Path,
    target_path: Path,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    error = _image_error_map(draft_path, target_path)
    ranked = []
    for index, stroke in enumerate(strokes):
        score = score_target_stroke(stroke, error)
        if score >= min_score:
            ranked.append({"source_index": index, "stroke": stroke, "score": score})
    ranked.sort(key=lambda entry: (float(entry["score"]), -int(entry["source_index"])), reverse=True)
    return ranked


def score_target_stroke(stroke: dict[str, Any], error_map: np.ndarray) -> float:
    height, width = error_map.shape
    x = min(width - 1, max(0, int(float(stroke["x"]) * width)))
    y = min(height - 1, max(0, int(float(stroke["y"]) * height)))
    length_pixels = max(1.0, float(stroke["length"]) * width)
    width_pixels = max(1.0, float(stroke["width"]) * width)
    radius = max(2, int(math.ceil(max(length_pixels, width_pixels) * 0.75)))
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    local = error_map[top:bottom, left:right]
    local_mean = float(local.mean()) if local.size else 0.0
    local_max = float(local.max()) if local.size else 0.0
    center = float(error_map[y, x])
    area = max(1.0, length_pixels * width_pixels)
    area_score = min(8.0, math.sqrt(area))
    return (0.55 * local_mean + 0.30 * center + 0.15 * local_max) * area_score


def _image_error_map(draft_path: Path, target_path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(draft_path) as draft_image, Image.open(target_path) as target_image:
        draft = np.asarray(draft_image.convert("RGB"), dtype=np.float32) / 255.0
        target = np.asarray(target_image.convert("RGB"), dtype=np.float32) / 255.0
    if draft.shape != target.shape:
        raise ValueError("draft and target image sizes differ")
    return np.max(np.abs(target - draft), axis=2)


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
        **_average_structure_metrics(rendered),
        "checkpoint_status": "visual_pass" if visual_pass else "visual_failed",
        "status_histogram": _status_histogram(exported),
    }


def _average_structure_metrics(entries: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = ("masked_mad_improvement", "gradient_improvement", "edge_overlap", "outside_mask_change")
    totals = {name: 0.0 for name in metric_names}
    counts = {name: 0 for name in metric_names}
    for entry in entries:
        diagnostics_path = entry.get("diagnostics")
        if not diagnostics_path:
            continue
        diagnostics = _read_json(Path(diagnostics_path))
        structure = diagnostics.get("structure_metrics", {})
        for name in metric_names:
            value = structure.get(name)
            if isinstance(value, (int, float)):
                totals[name] += float(value)
                counts[name] += 1
    return {f"mean_{name}": totals[name] / counts[name] if counts[name] else 0.0 for name in metric_names}


def _status_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        histogram[status] = histogram.get(status, 0) + 1
    return dict(sorted(histogram.items()))


def _validate_config(config: TargetStrokeRetrievalConfig) -> None:
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.recursive_passes <= 0:
        raise ValueError("recursive_passes must be positive")
    if config.strokes_per_pass <= 0:
        raise ValueError("strokes_per_pass must be positive")
    if config.min_pass_mad_improvement < 0.0:
        raise ValueError("min_pass_mad_improvement must be non-negative")
    if config.stroke_source not in (STROKE_SOURCE_FINISHING, STROKE_SOURCE_FULL_PROGRAM):
        raise ValueError(f"stroke_source must be {STROKE_SOURCE_FINISHING!r} or {STROKE_SOURCE_FULL_PROGRAM!r}")
    if config.target_mode not in (TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE):
        raise ValueError(
            "target_mode must be "
            f"{TARGET_MODE_TARGET_IMAGE!r}, {TARGET_MODE_FINISHED_IMAGE!r}, or {TARGET_MODE_SOURCE_IMAGE!r}"
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export V6 target-stroke retrieval oracle predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--recursive-passes", type=int, default=DEFAULT_RECURSIVE_PASSES)
    parser.add_argument("--strokes-per-pass", type=int, default=DEFAULT_STROKES_PER_PASS)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-pass-mad-improvement", type=float, default=DEFAULT_MIN_PASS_MAD_IMPROVEMENT)
    parser.add_argument("--stroke-source", choices=(STROKE_SOURCE_FINISHING, STROKE_SOURCE_FULL_PROGRAM), default=DEFAULT_STROKE_SOURCE)
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_TARGET_IMAGE, TARGET_MODE_FINISHED_IMAGE, TARGET_MODE_SOURCE_IMAGE),
        default=DEFAULT_TARGET_MODE,
    )
    parser.add_argument("--keep-empty-passes", action="store_true")
    parser.add_argument("--require-target-contract", default=DEFAULT_TARGET_CONTRACT)
    parser.add_argument("--min-changed-pixel-ratio", type=float, default=DEFAULT_MIN_CHANGED_PIXEL_RATIO)
    parser.add_argument("--min-gradient-improvement", type=float, default=DEFAULT_MIN_GRADIENT_IMPROVEMENT)
    parser.add_argument("--min-edge-overlap", type=float, default=DEFAULT_MIN_EDGE_OVERLAP)
    parser.add_argument("--no-render", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
