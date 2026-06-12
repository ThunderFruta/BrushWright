"""Export rendered predictions from the visual-delta stroke compiler."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch

from Source.Model.export_test_predictions import _brush_from_id, _program_like, _write_comparison_strip
from Source.Model.paint_transformer_soft_renderer import render_paint_transformer_soft_strokes
from Source.Model.prediction_diagnostics import compute_prediction_diagnostics
from Source.Model.stroke_dataset import DEFAULT_IMAGE_SIZE, load_draft_image_tensor
from Source.Model.train_strokes import DEFAULT_CUDA_ATTENTION_BACKEND, _configure_cuda_attention, _resolve_device
from Source.Model.visual_delta_dataset import (
    TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET,
    VisualDeltaBatch,
    VisualDeltaStrokeDataset,
    _build_edit_mask,
    _patch_offsets,
    _stroke_inside_patch,
    _stroke_to_patch_numeric,
    collate_visual_delta_patches,
    patch_numeric_to_global_stroke,
    visual_delta_batch_to_device,
)
from Source.Model.visual_delta_predictor import DEFAULT_MAX_STROKES, VisualDeltaStrokeCompiler, VisualDeltaStrokeCompilerConfig
from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_CHECKPOINT = Path("Models/Checkpoints/VisualDeltaStrokeCompilerV8UsableV1Large/best.pt")
DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/VisualDeltaPredictionsV8UsableV1Large")
DEFAULT_SPLIT = "Test"
DEFAULT_LIMIT = 4
DEFAULT_MAX_STROKES_PER_SAMPLE = DEFAULT_MAX_STROKES
DEFAULT_MAX_STROKES_PER_PATCH = DEFAULT_MAX_STROKES
DEFAULT_MIN_RENDER_AREA = 8.0
DEFAULT_RECURSIVE_PASSES = 1
DEFAULT_MIN_PASS_MAD_IMPROVEMENT = 0.1
DEFAULT_PRESENT_THRESHOLD = 0.05
RANKING_MODE_VISUAL_DELTA = "visual-delta"


@dataclass(frozen=True)
class ExportVisualDeltaConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint: Path = DEFAULT_CHECKPOINT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    sample_id: str | None = None
    limit: int = DEFAULT_LIMIT
    device: str = "auto"
    cuda_attention_backend: str = DEFAULT_CUDA_ATTENTION_BACKEND
    min_changed_pixel_ratio: float = 0.005
    min_gradient_improvement: float = 0.0
    min_edge_overlap: float = 0.02
    present_threshold: float = DEFAULT_PRESENT_THRESHOLD
    min_export_candidates_per_sample: int = 0
    max_strokes_per_sample: int = DEFAULT_MAX_STROKES_PER_SAMPLE
    max_strokes_per_patch: int = DEFAULT_MAX_STROKES_PER_PATCH
    min_render_area: float = DEFAULT_MIN_RENDER_AREA
    ranking_mode: str = RANKING_MODE_VISUAL_DELTA
    require_target_contract: str | None = TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET
    recursive_passes: int = DEFAULT_RECURSIVE_PASSES
    strokes_per_pass: int = DEFAULT_MAX_STROKES_PER_SAMPLE
    stop_on_non_improvement: bool = False
    min_pass_mad_improvement: float = DEFAULT_MIN_PASS_MAD_IMPROVEMENT
    keep_structure_failed_passes: bool = True
    render: bool = True
    allow_visual_failed_checkpoint: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exported = export_visual_delta_predictions(
        ExportVisualDeltaConfig(
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            split=args.split,
            sample_id=args.sample_id,
            limit=args.limit,
            device=args.device,
            cuda_attention_backend=args.cuda_attention_backend,
            min_changed_pixel_ratio=args.min_changed_pixel_ratio,
            min_gradient_improvement=args.min_gradient_improvement,
            min_edge_overlap=args.min_edge_overlap,
            present_threshold=args.present_threshold,
            min_export_candidates_per_sample=args.min_export_candidates_per_sample,
            max_strokes_per_sample=args.max_strokes_per_sample,
            max_strokes_per_patch=args.max_strokes_per_patch,
            min_render_area=args.min_render_area,
            ranking_mode=args.ranking_mode,
            require_target_contract=args.require_target_contract,
            recursive_passes=args.recursive_passes,
            strokes_per_pass=args.strokes_per_pass,
            stop_on_non_improvement=args.stop_on_non_improvement,
            min_pass_mad_improvement=args.min_pass_mad_improvement,
            keep_structure_failed_passes=args.keep_structure_failed_passes,
            render=not args.no_render,
            allow_visual_failed_checkpoint=args.allow_visual_failed_checkpoint,
        )
    )
    print(json.dumps({"exported": exported}, indent=2), flush=True)
    return 0


def export_visual_delta_predictions(config: ExportVisualDeltaConfig) -> list[dict[str, Any]]:
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.max_strokes_per_sample <= 0:
        raise ValueError("max_strokes_per_sample must be positive")
    if config.max_strokes_per_patch <= 0:
        raise ValueError("max_strokes_per_patch must be positive")
    if config.min_export_candidates_per_sample < 0:
        raise ValueError("min_export_candidates_per_sample must be non-negative")
    if config.recursive_passes <= 0:
        raise ValueError("recursive_passes must be positive")
    if config.strokes_per_pass <= 0:
        raise ValueError("strokes_per_pass must be positive")
    if config.min_pass_mad_improvement < 0.0:
        raise ValueError("min_pass_mad_improvement must be non-negative")
    if config.min_render_area < 0.0:
        raise ValueError("min_render_area must be non-negative")
    if config.ranking_mode != RANKING_MODE_VISUAL_DELTA:
        raise ValueError(f"ranking_mode must be {RANKING_MODE_VISUAL_DELTA!r}")
    _configure_cuda_attention(config.cuda_attention_backend)
    device = _resolve_device(config.device)
    checkpoint_path = _resolve_checkpoint_path(config.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    _validate_export_checkpoint(checkpoint, checkpoint_path, allow_visual_failed=config.allow_visual_failed_checkpoint)
    model_config = VisualDeltaStrokeCompilerConfig(**checkpoint["model_config"])
    dataset_config = checkpoint.get("dataset_config", {})
    model = VisualDeltaStrokeCompiler(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_root = config.data_root / config.split
    manifest = _read_json(split_root / "dataset_manifest.json")
    manifest_samples = list(manifest.get("samples", []))
    if config.sample_id is not None:
        samples = [entry for entry in manifest_samples if str(entry.get("sample_id")) == config.sample_id]
        if not samples:
            raise ValueError(f"sample_id not found in {config.split}: {config.sample_id}")
    else:
        samples = manifest_samples[: config.limit]
    dataset = VisualDeltaStrokeDataset(
        split_root,
        patch_size=int(dataset_config.get("patch_size", 64)),
        patch_stride=int(dataset_config.get("patch_stride", 64)),
        max_strokes_per_patch=int(dataset_config.get("max_strokes_per_patch", model_config.max_strokes)),
        mask_threshold=float(dataset_config.get("mask_threshold", 0.04)),
        min_changed_pixels=int(dataset_config.get("min_changed_pixels", 16)),
        negative_patch_ratio=0.0,
        require_structure_targets=bool(dataset_config.get("require_structure_targets", False)),
        require_target_contract=dataset_config.get("require_target_contract", config.require_target_contract),
    )

    prepared_root = prepare_latest_output_root(config.output_root)
    output_root = prepared_root / config.split
    output_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, sample_entry in enumerate(samples, start=1):
        sample_dir = split_root / sample_entry["path"]
        sample = _read_json(sample_dir / "sample.json")
        output_dir = output_root / str(sample["sample_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(samples)}] exporting {sample['sample_id']} -> {output_dir}", flush=True)
        exported.append(
            _export_sample(
                dataset=dataset,
                sample=sample,
                sample_dir=sample_dir,
                output_dir=output_dir,
                model=model,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                device=device,
                present_threshold=config.present_threshold,
                min_export_candidates_per_sample=config.min_export_candidates_per_sample,
                max_strokes_per_sample=config.max_strokes_per_sample,
                max_strokes_per_patch=config.max_strokes_per_patch,
                min_render_area=config.min_render_area,
                ranking_mode=config.ranking_mode,
                recursive_passes=config.recursive_passes,
                strokes_per_pass=config.strokes_per_pass,
                stop_on_non_improvement=config.stop_on_non_improvement,
                min_pass_mad_improvement=config.min_pass_mad_improvement,
                keep_structure_failed_passes=config.keep_structure_failed_passes,
                min_changed_pixel_ratio=config.min_changed_pixel_ratio,
                min_gradient_improvement=config.min_gradient_improvement,
                min_edge_overlap=config.min_edge_overlap,
                render=config.render,
            )
        )
    _write_json(
        prepared_root / "export_manifest.json",
        {"version": 1, "split": config.split, "summary": _export_summary(exported), "samples": exported},
    )
    return exported


def _export_sample(
    dataset: VisualDeltaStrokeDataset,
    sample: dict[str, Any],
    sample_dir: Path,
    output_dir: Path,
    model: VisualDeltaStrokeCompiler,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    present_threshold: float,
    min_export_candidates_per_sample: int,
    max_strokes_per_sample: int,
    max_strokes_per_patch: int,
    min_render_area: float,
    ranking_mode: str,
    recursive_passes: int,
    strokes_per_pass: int,
    stop_on_non_improvement: bool,
    min_pass_mad_improvement: float,
    keep_structure_failed_passes: bool,
    min_changed_pixel_ratio: float,
    min_gradient_improvement: float,
    min_edge_overlap: float,
    render: bool,
) -> dict[str, Any]:
    if recursive_passes == 1:
        return _export_single_pass_sample(
            dataset=dataset,
            sample=sample,
            sample_dir=sample_dir,
            output_dir=output_dir,
            model=model,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            device=device,
            present_threshold=present_threshold,
            min_export_candidates_per_sample=min_export_candidates_per_sample,
            max_strokes_per_sample=max_strokes_per_sample,
            max_strokes_per_patch=max_strokes_per_patch,
            min_render_area=min_render_area,
            ranking_mode=ranking_mode,
            min_changed_pixel_ratio=min_changed_pixel_ratio,
            min_gradient_improvement=min_gradient_improvement,
            min_edge_overlap=min_edge_overlap,
            render=render,
        )
    return _export_recursive_sample(
        dataset=dataset,
        sample=sample,
        sample_dir=sample_dir,
        output_dir=output_dir,
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
        present_threshold=present_threshold,
        min_export_candidates_per_sample=min_export_candidates_per_sample,
        strokes_per_pass=strokes_per_pass,
        max_strokes_per_patch=max_strokes_per_patch,
        min_render_area=min_render_area,
        ranking_mode=ranking_mode,
        recursive_passes=recursive_passes,
        stop_on_non_improvement=stop_on_non_improvement,
        min_pass_mad_improvement=min_pass_mad_improvement,
        keep_structure_failed_passes=keep_structure_failed_passes,
        min_changed_pixel_ratio=min_changed_pixel_ratio,
        min_gradient_improvement=min_gradient_improvement,
        min_edge_overlap=min_edge_overlap,
        render=render,
    )


def _export_single_pass_sample(
    dataset: VisualDeltaStrokeDataset,
    sample: dict[str, Any],
    sample_dir: Path,
    output_dir: Path,
    model: VisualDeltaStrokeCompiler,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    present_threshold: float,
    min_export_candidates_per_sample: int,
    max_strokes_per_sample: int,
    max_strokes_per_patch: int,
    min_render_area: float,
    ranking_mode: str,
    min_changed_pixel_ratio: float,
    min_gradient_improvement: float,
    min_edge_overlap: float,
    render: bool,
) -> dict[str, Any]:
    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    patch_indices = [
        patch_index
        for patch_index, patch in enumerate(dataset.patch_index)
        if patch.sample_id == str(sample["sample_id"]) and patch.changed
    ]
    threshold_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    scored_candidate_count = 0
    present_score_total = 0.0
    present_score_count = 0
    max_present = 0.0
    order = 0
    with torch.no_grad():
        for patch_index in patch_indices:
            item = dataset[patch_index]
            batch = visual_delta_batch_to_device(collate_visual_delta_patches([item]), device)
            output = model(batch)
            present = torch.sigmoid(output.pred_present_logits[0]).detach().cpu()
            numeric = output.pred_numeric[0].detach().cpu()
            brush_logits = output.pred_brush_logits[0].detach().cpu()
            patch_bounds = item.patch_bounds.tolist()
            patch_threshold_candidates: list[dict[str, Any]] = []
            for slot, present_score in enumerate(present.tolist()):
                present_score = float(present_score)
                present_score_total += present_score
                present_score_count += 1
                max_present = max(max_present, present_score)
                brush_id = int(torch.argmax(brush_logits[slot]).item())
                brush = _brush_from_id(brush_id, checkpoint)
                candidate_numeric = _candidate_numeric_with_residual_color(numeric[slot], item.patch_tensor)
                score, area_pixels = _visual_delta_candidate_score(
                    candidate_numeric,
                    present_score=present_score,
                    patch_tensor=item.patch_tensor,
                    min_render_area=min_render_area,
                    ranking_mode=ranking_mode,
                )
                if score <= 0.0 or area_pixels < min_render_area:
                    continue
                scored_candidate_count += 1
                candidate = {
                    "stroke": patch_numeric_to_global_stroke(candidate_numeric.tolist(), brush, patch_bounds),
                    "score": score,
                    "area_pixels": area_pixels,
                    "present_score": present_score,
                    "passed_present_threshold": present_score >= present_threshold,
                    "order": order,
                }
                order += 1
                if candidate["passed_present_threshold"]:
                    patch_threshold_candidates.append(candidate)
                else:
                    fallback_candidates.append(candidate)
            patch_threshold_candidates.sort(key=lambda candidate: (candidate["score"], candidate["present_score"]), reverse=True)
            threshold_candidates.extend(patch_threshold_candidates[:max_strokes_per_patch])
    predicted_candidates = _apply_export_candidate_fallback(
        threshold_candidates=threshold_candidates,
        fallback_candidates=fallback_candidates,
        min_export_candidates_per_sample=min_export_candidates_per_sample,
    )
    selected_candidates = _select_ranked_export_candidates(predicted_candidates, max_strokes_per_sample)
    predicted_strokes = [candidate["stroke"] for candidate in selected_candidates]
    fallback_added_count = sum(1 for candidate in predicted_candidates if not candidate.get("passed_present_threshold", True))

    added_metadata = {
        **dict(target_program.get("metadata", {})),
        "prediction_source": str(checkpoint_path),
        "sample_id": sample["sample_id"],
        "split": "visual_delta_added_strokes",
        "export_filter": {
            "present_threshold": present_threshold,
            "min_export_candidates_per_sample": min_export_candidates_per_sample,
            "max_strokes_per_sample": max_strokes_per_sample,
            "max_strokes_per_patch": max_strokes_per_patch,
            "min_render_area": min_render_area,
            "ranking_mode": ranking_mode,
            "max_present": max_present,
            "mean_present": present_score_total / present_score_count if present_score_count else 0.0,
            "present_score_count": present_score_count,
            "candidate_count_before_threshold": scored_candidate_count,
            "candidate_count_after_threshold": len(threshold_candidates),
            "fallback_candidate_count": fallback_added_count,
            "candidate_count": len(predicted_candidates),
            "selected_count": len(predicted_strokes),
        },
    }
    predicted_finishing_program = _program_without_validation(
        template=target_program,
        metadata=added_metadata,
        strokes=predicted_strokes,
    )
    predicted_full_program = _program_like(
        template=base_program,
        metadata={
            **dict(base_program.get("metadata", {})),
            "prediction_source": str(checkpoint_path),
            "sample_id": sample["sample_id"],
            "split": "base_plus_visual_delta_strokes",
        },
        strokes=[*base_program["strokes"], *predicted_strokes],
    )
    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(output_dir / "sample.json", sample | {"prediction_checkpoint": str(checkpoint_path)})
    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    shutil.copy2(_target_image_path(sample_dir, sample), output_dir / "target.png")

    if render:
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
            target_strokes=target_program["strokes"],
            min_changed_pixel_ratio=min_changed_pixel_ratio,
            min_gradient_improvement=min_gradient_improvement,
            min_edge_overlap=min_edge_overlap,
        )
        _write_json(output_dir / "diagnostics.json", diagnostics)
    else:
        diagnostics = {"status": "not_rendered", "visual_improved": None}

    return {
        "sample_id": sample["sample_id"],
        "output_dir": str(output_dir),
        "draft": str(output_dir / "draft.png"),
        "target": str(output_dir / "target.png"),
        "predicted": str(output_dir / "predicted.png") if render else None,
        "comparison": str(output_dir / "comparison.png") if render else None,
        "diagnostics": str(output_dir / "diagnostics.json") if render else None,
        "status": diagnostics["status"],
        "visual_improved": diagnostics["visual_improved"],
        "added_strokes": str(output_dir / "added_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
    }


def _export_recursive_sample(
    dataset: VisualDeltaStrokeDataset,
    sample: dict[str, Any],
    sample_dir: Path,
    output_dir: Path,
    model: VisualDeltaStrokeCompiler,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    present_threshold: float,
    min_export_candidates_per_sample: int,
    strokes_per_pass: int,
    max_strokes_per_patch: int,
    min_render_area: float,
    ranking_mode: str,
    recursive_passes: int,
    stop_on_non_improvement: bool,
    min_pass_mad_improvement: float,
    keep_structure_failed_passes: bool,
    min_changed_pixel_ratio: float,
    min_gradient_improvement: float,
    min_edge_overlap: float,
    render: bool,
) -> dict[str, Any]:
    if not render:
        raise ValueError("recursive export requires rendering")
    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    shutil.copy2(_target_image_path(sample_dir, sample), output_dir / "target.png")

    current_draft_path = output_dir / "draft.png"
    all_strokes: list[dict[str, Any]] = []
    stroke_keys: set[tuple[Any, ...]] = set()
    pass_entries: list[dict[str, Any]] = []
    stop_reason = "max_passes"
    previous_mad = _diagnostic_mad(current_draft_path, output_dir / "target.png")

    for pass_index in range(1, recursive_passes + 1):
        pass_dir = output_dir / f"pass_{pass_index:04d}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_draft_path, pass_dir / "draft.png")
        shutil.copy2(output_dir / "target.png", pass_dir / "target.png")
        pass_candidates = _predict_sample_candidates(
            dataset=dataset,
            sample=sample,
            sample_dir=sample_dir,
            current_draft_path=current_draft_path,
            target_path=output_dir / "target.png",
            model=model,
            checkpoint=checkpoint,
            device=device,
            present_threshold=present_threshold,
            min_export_candidates_per_sample=min_export_candidates_per_sample,
            max_strokes_per_patch=max_strokes_per_patch,
            min_render_area=min_render_area,
            ranking_mode=ranking_mode,
            stroke_keys=stroke_keys,
        )
        selected_candidates = _select_ranked_export_candidates(pass_candidates, strokes_per_pass)
        pass_strokes = [candidate["stroke"] for candidate in selected_candidates]

        pass_program = _program_without_validation(
            template=target_program,
            metadata={
                **dict(target_program.get("metadata", {})),
                "prediction_source": str(checkpoint_path),
                "sample_id": sample["sample_id"],
                "split": "visual_delta_recursive_pass_strokes",
                "recursive_pass": pass_index,
                "export_filter": {
                    "present_threshold": present_threshold,
                    "min_export_candidates_per_sample": min_export_candidates_per_sample,
                    "strokes_per_pass": strokes_per_pass,
                    "max_strokes_per_patch": max_strokes_per_patch,
                    "min_render_area": min_render_area,
                    "ranking_mode": ranking_mode,
                    "candidate_count": len(pass_candidates),
                    "selected_count": len(pass_strokes),
                    "duplicate_suppression": True,
                },
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
            target_strokes=target_program["strokes"],
            min_changed_pixel_ratio=min_changed_pixel_ratio,
            min_gradient_improvement=min_gradient_improvement,
            min_edge_overlap=min_edge_overlap,
        )
        current_mad = diagnostics["image_deltas"]["predicted_to_target"]["mean_absolute_difference"]
        pass_improvement = previous_mad - current_mad
        accepted = True
        diagnostics["recursive_pass"] = {
            "pass_index": pass_index,
            "previous_predicted_to_target_mad": previous_mad,
            "current_predicted_to_target_mad": current_mad,
            "pass_mad_improvement": pass_improvement,
            "accepted": accepted,
        }
        if diagnostics["status"] == "failed_structure_noise" and not keep_structure_failed_passes:
            accepted = False
            stop_reason = "structure_failed"
        elif diagnostics["status"] == "failed_low_pixel_change":
            accepted = False
            stop_reason = "low_pixel_change"
        elif stop_on_non_improvement and pass_improvement < min_pass_mad_improvement:
            accepted = False
            stop_reason = "non_improvement"
        diagnostics["recursive_pass"]["accepted"] = accepted
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
                "pass_mad_improvement": pass_improvement,
                "accepted": accepted,
            }
        )
        if accepted:
            _record_stroke_keys(pass_strokes, stroke_keys)
            all_strokes.extend(pass_strokes)
            current_draft_path = pass_dir / "predicted.png"
            previous_mad = current_mad
        if stop_reason != "max_passes":
            break

    added_metadata = {
        **dict(target_program.get("metadata", {})),
        "prediction_source": str(checkpoint_path),
        "sample_id": sample["sample_id"],
        "split": "visual_delta_recursive_added_strokes",
        "recursive_export": {
            "recursive_passes_requested": recursive_passes,
            "recursive_passes_completed": len(pass_entries),
            "strokes_per_pass": strokes_per_pass,
            "stop_reason": stop_reason,
            "stop_on_non_improvement": stop_on_non_improvement,
            "min_pass_mad_improvement": min_pass_mad_improvement,
            "keep_structure_failed_passes": keep_structure_failed_passes,
            "duplicate_suppression": True,
        },
    }
    predicted_finishing_program = _program_without_validation(
        template=target_program,
        metadata=added_metadata,
        strokes=all_strokes,
    )
    predicted_full_program = _program_like(
        template=base_program,
        metadata={
            **dict(base_program.get("metadata", {})),
            "prediction_source": str(checkpoint_path),
            "sample_id": sample["sample_id"],
            "split": "base_plus_recursive_visual_delta_strokes",
        },
        strokes=[*base_program["strokes"], *all_strokes],
    )
    _write_json(output_dir / "added_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(output_dir / "sample.json", sample | {"prediction_checkpoint": str(checkpoint_path)})
    shutil.copy2(current_draft_path, output_dir / "predicted.png")
    _write_comparison_strip(
        output_dir / "draft.png",
        output_dir / "target.png",
        output_dir / "predicted.png",
        output_dir / "comparison.png",
    )
    final_diagnostics = compute_prediction_diagnostics(
        draft_path=output_dir / "draft.png",
        target_path=output_dir / "target.png",
        predicted_path=output_dir / "predicted.png",
        predicted_strokes=all_strokes,
        target_strokes=target_program["strokes"],
        min_changed_pixel_ratio=min_changed_pixel_ratio,
        min_gradient_improvement=min_gradient_improvement,
        min_edge_overlap=min_edge_overlap,
    )
    final_diagnostics["recursive_export"] = {
        "passes": pass_entries,
        "stop_reason": stop_reason,
        "total_added_strokes": len(all_strokes),
        "max_possible_added_strokes": recursive_passes * strokes_per_pass,
    }
    _write_json(output_dir / "diagnostics.json", final_diagnostics)
    _write_json(
        output_dir / "recursive_manifest.json",
        {
            "version": 1,
            "sample_id": sample["sample_id"],
            "prediction_source": str(checkpoint_path),
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
        "recursive_manifest": str(output_dir / "recursive_manifest.json"),
        "recursive_passes_completed": len(pass_entries),
        "recursive_stop_reason": stop_reason,
        "added_stroke_count": len(all_strokes),
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
        **_average_structure_metrics(rendered),
        "checkpoint_status": "visual_pass" if visual_pass else "visual_failed",
        "status_histogram": _status_histogram(exported),
        **_average_export_filter_metrics(exported),
    }


def _predict_sample_candidates(
    dataset: VisualDeltaStrokeDataset,
    sample: dict[str, Any],
    sample_dir: Path,
    current_draft_path: Path,
    target_path: Path,
    model: VisualDeltaStrokeCompiler,
    checkpoint: dict[str, Any],
    device: torch.device,
    present_threshold: float,
    min_export_candidates_per_sample: int,
    max_strokes_per_patch: int,
    min_render_area: float,
    ranking_mode: str,
    stroke_keys: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    draft_image = load_draft_image_tensor(current_draft_path)
    target_image = load_draft_image_tensor(target_path)
    error_map = torch.abs(target_image - draft_image)
    edit_mask = _build_edit_mask(error_map, dataset.mask_threshold)
    finishing_program = _read_json(sample_dir / sample["finishing_strokes"])
    finishing_strokes = finishing_program["strokes"]
    threshold_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    order = 0
    with torch.no_grad():
        for top in _patch_offsets(DEFAULT_IMAGE_SIZE, dataset.patch_size, dataset.patch_stride):
            for left in _patch_offsets(DEFAULT_IMAGE_SIZE, dataset.patch_size, dataset.patch_stride):
                mask_patch = edit_mask[:, top : top + dataset.patch_size, left : left + dataset.patch_size]
                if int(mask_patch.sum().item()) < dataset.min_changed_pixels:
                    continue
                patch_tensor = _runtime_patch_tensor(
                    draft_image=draft_image,
                    target_image=target_image,
                    error_map=error_map,
                    edit_mask=edit_mask,
                    left=left,
                    top=top,
                    patch_size=dataset.patch_size,
                )
                batch = _runtime_visual_delta_batch(
                    patch_tensor=patch_tensor,
                    finishing_strokes=finishing_strokes,
                    left=left,
                    top=top,
                    patch_size=dataset.patch_size,
                    max_strokes_per_patch=dataset.max_strokes_per_patch,
                    device=device,
                )
                output = model(batch)
                present = torch.sigmoid(output.pred_present_logits[0]).detach().cpu()
                numeric = output.pred_numeric[0].detach().cpu()
                brush_logits = output.pred_brush_logits[0].detach().cpu()
                patch_bounds = [
                    left / DEFAULT_IMAGE_SIZE,
                    top / DEFAULT_IMAGE_SIZE,
                    (left + dataset.patch_size) / DEFAULT_IMAGE_SIZE,
                    (top + dataset.patch_size) / DEFAULT_IMAGE_SIZE,
                ]
                patch_threshold_candidates: list[dict[str, Any]] = []
                for slot, present_score in enumerate(present.tolist()):
                    present_score = float(present_score)
                    brush_id = int(torch.argmax(brush_logits[slot]).item())
                    brush = _brush_from_id(brush_id, checkpoint)
                    candidate_numeric = _candidate_numeric_with_residual_color(numeric[slot], patch_tensor)
                    score, area_pixels = _visual_delta_candidate_score(
                        candidate_numeric,
                        present_score=present_score,
                        patch_tensor=patch_tensor,
                        min_render_area=min_render_area,
                        ranking_mode=ranking_mode,
                    )
                    if score <= 0.0 or area_pixels < min_render_area:
                        continue
                    stroke = patch_numeric_to_global_stroke(candidate_numeric.tolist(), brush, patch_bounds)
                    if _stroke_key(stroke) in stroke_keys:
                        continue
                    candidate = {
                        "stroke": stroke,
                        "score": score,
                        "area_pixels": area_pixels,
                        "present_score": present_score,
                        "passed_present_threshold": present_score >= present_threshold,
                        "order": order,
                    }
                    order += 1
                    if candidate["passed_present_threshold"]:
                        patch_threshold_candidates.append(candidate)
                    else:
                        fallback_candidates.append(candidate)
                patch_threshold_candidates.sort(key=lambda candidate: (candidate["score"], candidate["present_score"]), reverse=True)
                threshold_candidates.extend(patch_threshold_candidates[:max_strokes_per_patch])
    return _apply_export_candidate_fallback(
        threshold_candidates=threshold_candidates,
        fallback_candidates=fallback_candidates,
        min_export_candidates_per_sample=min_export_candidates_per_sample,
    )


def _runtime_patch_tensor(
    draft_image: torch.Tensor,
    target_image: torch.Tensor,
    error_map: torch.Tensor,
    edit_mask: torch.Tensor,
    left: int,
    top: int,
    patch_size: int,
) -> torch.Tensor:
    right = left + patch_size
    bottom = top + patch_size
    return torch.cat(
        [
            draft_image[:, top:bottom, left:right],
            target_image[:, top:bottom, left:right],
            error_map[:, top:bottom, left:right],
            edit_mask[:, top:bottom, left:right],
        ],
        dim=0,
    )


def _runtime_visual_delta_batch(
    patch_tensor: torch.Tensor,
    finishing_strokes: list[dict[str, Any]],
    left: int,
    top: int,
    patch_size: int,
    max_strokes_per_patch: int,
    device: torch.device,
) -> VisualDeltaBatch:
    numeric = torch.zeros(max_strokes_per_patch, 9, dtype=torch.float32)
    brush_ids = torch.zeros(max_strokes_per_patch, dtype=torch.long)
    present = torch.zeros(max_strokes_per_patch, dtype=torch.float32)
    padding_mask = torch.ones(max_strokes_per_patch, dtype=torch.bool)
    selected = [
        stroke
        for stroke in finishing_strokes
        if _stroke_inside_patch(stroke, left=left, top=top, patch_size=patch_size)
    ][:max_strokes_per_patch]
    for index, stroke in enumerate(selected):
        numeric[index] = torch.tensor(_stroke_to_patch_numeric(stroke, left=left, top=top, patch_size=patch_size))
        present[index] = 1.0
        padding_mask[index] = False
    patch_bounds = torch.tensor(
        [[left, top, left + patch_size, top + patch_size]],
        dtype=torch.float32,
    ) / DEFAULT_IMAGE_SIZE
    batch = VisualDeltaBatch(
        patch_tensor=patch_tensor.unsqueeze(0),
        target_numeric=numeric.unsqueeze(0),
        target_brush_ids=brush_ids.unsqueeze(0),
        target_present=present.unsqueeze(0),
        target_padding_mask=padding_mask.unsqueeze(0),
        sample_ids=("runtime",),
        patch_bounds=patch_bounds,
        changed=torch.tensor([True], dtype=torch.bool),
    )
    return visual_delta_batch_to_device(batch, device)


def _record_stroke_keys(strokes: list[dict[str, Any]], stroke_keys: set[tuple[Any, ...]]) -> None:
    for stroke in strokes:
        stroke_keys.add(_stroke_key(stroke))


def _stroke_key(stroke: dict[str, Any]) -> tuple[Any, ...]:
    color = stroke.get("color", [0.0, 0.0, 0.0])
    return (
        round(float(stroke.get("x", 0.0)), 4),
        round(float(stroke.get("y", 0.0)), 4),
        round(float(stroke.get("angle", 0.0)), 3),
        round(float(stroke.get("length", 0.0)), 4),
        round(float(stroke.get("width", 0.0)), 4),
        tuple(round(float(channel), 3) for channel in color),
        str(stroke.get("brush", "")),
    )


def _diagnostic_mad(first_path: Path, second_path: Path) -> float:
    diagnostics = compute_prediction_diagnostics(
        draft_path=first_path,
        target_path=second_path,
        predicted_path=first_path,
        predicted_strokes=[],
        target_strokes=[],
    )
    return diagnostics["image_deltas"]["draft_to_target"]["mean_absolute_difference"]


def _status_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        histogram[status] = histogram.get(status, 0) + 1
    return dict(sorted(histogram.items()))


def _average_structure_metrics(entries: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = (
        "masked_mad_improvement",
        "gradient_improvement",
        "edge_overlap",
        "outside_mask_change",
    )
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
    return {
        f"mean_{name}": totals[name] / counts[name] if counts[name] else 0.0
        for name in metric_names
    }


def _average_export_filter_metrics(entries: list[dict[str, Any]]) -> dict[str, float]:
    sum_metric_names = (
        "candidate_count_before_threshold",
        "candidate_count_after_threshold",
        "candidate_count",
        "selected_count",
        "fallback_candidate_count",
        "present_score_count",
    )
    sum_totals = {name: 0.0 for name in sum_metric_names}
    max_present = 0.0
    weighted_mean_present_total = 0.0
    weighted_mean_present_count = 0.0
    filter_count = 0
    for entry in entries:
        added_strokes_path = entry.get("added_strokes")
        if not added_strokes_path:
            continue
        try:
            program = _read_json(Path(added_strokes_path))
        except OSError:
            continue
        export_filter = program.get("metadata", {}).get("export_filter", {})
        if not isinstance(export_filter, dict):
            continue
        filter_count += 1
        for name in sum_metric_names:
            value = export_filter.get(name)
            if isinstance(value, (int, float)):
                sum_totals[name] += float(value)
        if isinstance(export_filter.get("max_present"), (int, float)):
            max_present = max(max_present, float(export_filter["max_present"]))
        mean_present = export_filter.get("mean_present")
        present_score_count = export_filter.get("present_score_count")
        if isinstance(mean_present, (int, float)) and isinstance(present_score_count, (int, float)):
            weighted_mean_present_total += float(mean_present) * float(present_score_count)
            weighted_mean_present_count += float(present_score_count)
    return {
        "max_present": max_present,
        "mean_present": weighted_mean_present_total / weighted_mean_present_count if weighted_mean_present_count else 0.0,
        "candidate_count_before_threshold": sum_totals["candidate_count_before_threshold"],
        "candidate_count_after_threshold": sum_totals["candidate_count_after_threshold"],
        "candidate_count": sum_totals["candidate_count"],
        "selected_count": sum_totals["selected_count"],
        "fallback_candidate_count": sum_totals["fallback_candidate_count"],
        "present_score_count": sum_totals["present_score_count"],
        "export_filter_count": float(filter_count),
    }


def _program_without_validation(template: dict[str, Any], metadata: dict[str, Any], strokes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": template["version"],
        "canvas": template["canvas"],
        "metadata": metadata,
        "strokes": strokes,
    }


def _candidate_numeric_with_residual_color(numeric: torch.Tensor, patch_tensor: torch.Tensor) -> torch.Tensor:
    if patch_tensor.ndim != 3 or patch_tensor.shape[0] < 10:
        raise ValueError("patch_tensor must have shape [10, height, width]")
    candidate = numeric.detach().clone()
    height = int(patch_tensor.shape[-2])
    width = int(patch_tensor.shape[-1])
    device = patch_tensor.device
    dtype = patch_tensor.dtype
    center_x = candidate[0].clamp(0.0, 1.0).to(device=device, dtype=dtype) * float(max(width - 1, 1))
    center_y = candidate[1].clamp(0.0, 1.0).to(device=device, dtype=dtype) * float(max(height - 1, 1))
    angle = candidate[2].to(device=device, dtype=dtype) * torch.pi
    length_radius = (candidate[3].clamp_min(0.0).to(device=device, dtype=dtype) * float(width) * 0.5).clamp_min(1.0)
    width_radius = (candidate[4].clamp_min(0.0).to(device=device, dtype=dtype) * float(height) * 0.5).clamp_min(1.0)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = x_coords - center_x
    dy = y_coords - center_y
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    along = dx * cos_a + dy * sin_a
    across = -dx * sin_a + dy * cos_a
    footprint = ((along / length_radius) ** 2 + (across / width_radius) ** 2 <= 1.0).to(dtype)
    edit_mask = patch_tensor[9].to(dtype)
    weights = footprint * edit_mask
    if float(weights.sum().item()) < 1.0:
        x = min(width - 1, max(0, int(float(center_x.detach().cpu().item()))))
        y = min(height - 1, max(0, int(float(center_y.detach().cpu().item()))))
        color = patch_tensor[3:6, y, x]
    else:
        color = (patch_tensor[3:6].to(dtype) * weights.unsqueeze(0)).sum(dim=(1, 2)) / weights.sum().clamp_min(1.0)
    candidate[6:9] = color.to(device=candidate.device, dtype=candidate.dtype).clamp(0.0, 1.0)
    return candidate


def _visual_delta_candidate_score(
    numeric: torch.Tensor,
    present_score: float,
    patch_tensor: torch.Tensor,
    min_render_area: float,
    ranking_mode: str = RANKING_MODE_VISUAL_DELTA,
) -> tuple[float, float]:
    if ranking_mode != RANKING_MODE_VISUAL_DELTA:
        raise ValueError(f"ranking_mode must be {RANKING_MODE_VISUAL_DELTA!r}")
    if patch_tensor.ndim != 3 or patch_tensor.shape[0] < 10:
        raise ValueError("patch_tensor must have shape [10, height, width]")
    height = int(patch_tensor.shape[-2])
    width = int(patch_tensor.shape[-1])
    x = min(width - 1, max(0, int(float(numeric[0].clamp(0.0, 1.0)) * width)))
    y = min(height - 1, max(0, int(float(numeric[1].clamp(0.0, 1.0)) * height)))
    area_pixels = _stroke_render_area_pixels(numeric, height=height, width=width)
    if area_pixels < min_render_area:
        return 0.0, area_pixels
    edit_mask = float(patch_tensor[9, y, x].item())
    if edit_mask <= 0.0:
        return 0.0, area_pixels
    rendered_improvement = _rendered_masked_candidate_improvement(
        numeric=numeric,
        present_score=present_score,
        patch_tensor=patch_tensor,
    )
    if rendered_improvement <= 0.0:
        return 0.0, area_pixels
    area_score = min(2.0, area_pixels / max(min_render_area, 1.0))
    score = float(present_score) * rendered_improvement * area_score
    return score, area_pixels


def _select_ranked_export_candidates(
    candidates: list[dict[str, Any]],
    max_strokes_per_sample: int,
) -> list[dict[str, Any]]:
    if max_strokes_per_sample <= 0:
        raise ValueError("max_strokes_per_sample must be positive")
    ranked = sorted(
        candidates,
        key=lambda candidate: (float(candidate["score"]), float(candidate.get("present_score", 0.0))),
        reverse=True,
    )
    selected = ranked[:max_strokes_per_sample]
    return sorted(selected, key=lambda candidate: int(candidate.get("order", 0)))


def _apply_export_candidate_fallback(
    threshold_candidates: list[dict[str, Any]],
    fallback_candidates: list[dict[str, Any]],
    min_export_candidates_per_sample: int,
) -> list[dict[str, Any]]:
    if min_export_candidates_per_sample < 0:
        raise ValueError("min_export_candidates_per_sample must be non-negative")
    selected = list(threshold_candidates)
    needed = max(0, min_export_candidates_per_sample - len(selected))
    if needed <= 0:
        return selected
    ranked_fallback = sorted(
        fallback_candidates,
        key=lambda candidate: (float(candidate.get("score", 0.0)), float(candidate.get("present_score", 0.0))),
        reverse=True,
    )
    selected.extend(ranked_fallback[:needed])
    return selected


def _stroke_render_area_pixels(numeric: torch.Tensor, height: int, width: int) -> float:
    length = float(numeric[3].clamp_min(0.0).item())
    stroke_width = float(numeric[4].clamp_min(0.0).item())
    return length * stroke_width * float(height * width)


def _rendered_masked_candidate_improvement(
    numeric: torch.Tensor,
    present_score: float,
    patch_tensor: torch.Tensor,
    outside_penalty_weight: float = 0.25,
) -> float:
    draft = patch_tensor[0:3].unsqueeze(0)
    target = patch_tensor[3:6].unsqueeze(0)
    edit_mask = patch_tensor[9:10].unsqueeze(0)
    if float(edit_mask.sum().item()) <= 0.0:
        return 0.0
    candidate = numeric.to(dtype=draft.dtype, device=draft.device).view(1, 1, -1)
    present_logits = torch.full(
        (1, 1),
        _logit(float(present_score)),
        dtype=draft.dtype,
        device=draft.device,
    )
    predicted = render_paint_transformer_soft_strokes(draft, candidate, present_logits)
    before_error = torch.abs(draft - target)
    after_error = torch.abs(predicted - target)
    inside_denominator = edit_mask.sum().clamp_min(1.0) * draft.shape[1]
    inside_improvement = ((before_error - after_error) * edit_mask).sum() / inside_denominator
    outside_mask = 1.0 - edit_mask
    outside_denominator = outside_mask.sum().clamp_min(1.0) * draft.shape[1]
    outside_change = (torch.abs(predicted - draft) * outside_mask).sum() / outside_denominator
    score = inside_improvement - outside_change * outside_penalty_weight
    return max(0.0, float(score.detach().cpu().item()))


def _logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return torch.logit(torch.tensor(clipped)).item()


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    raise OSError(f"checkpoint does not exist: {path}")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise OSError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def _validate_export_checkpoint(checkpoint: dict[str, Any], path: Path, allow_visual_failed: bool) -> None:
    if allow_visual_failed:
        return
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != "best":
        raise ValueError(
            "refusing to export visual-delta checkpoint that has not passed visual validation: "
            f"{path} has checkpoint_type={checkpoint_type!r}. "
            "Use --allow-visual-failed-checkpoint only for debugging."
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_image_path(sample_dir: Path, sample: dict[str, Any]) -> Path:
    image_name = sample.get("target_image") or sample.get("finished_image")
    if not image_name:
        raise ValueError(f"{sample_dir} sample is missing target_image metadata")
    return sample_dir / str(image_name)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export visual-delta stroke compiler predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-attention-backend", choices=("math", "default"), default=DEFAULT_CUDA_ATTENTION_BACKEND)
    parser.add_argument("--min-changed-pixel-ratio", type=float, default=0.005)
    parser.add_argument("--min-gradient-improvement", type=float, default=0.0)
    parser.add_argument("--min-edge-overlap", type=float, default=0.02)
    parser.add_argument("--present-threshold", type=float, default=DEFAULT_PRESENT_THRESHOLD)
    parser.add_argument("--min-export-candidates-per-sample", type=int, default=0)
    parser.add_argument("--max-strokes-per-sample", type=int, default=DEFAULT_MAX_STROKES_PER_SAMPLE)
    parser.add_argument("--max-strokes-per-patch", type=int, default=DEFAULT_MAX_STROKES_PER_PATCH)
    parser.add_argument("--min-render-area", type=float, default=DEFAULT_MIN_RENDER_AREA)
    parser.add_argument("--ranking-mode", choices=(RANKING_MODE_VISUAL_DELTA,), default=RANKING_MODE_VISUAL_DELTA)
    parser.add_argument("--require-target-contract", default=TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET)
    parser.add_argument("--recursive-passes", type=int, default=DEFAULT_RECURSIVE_PASSES)
    parser.add_argument("--strokes-per-pass", type=int, default=DEFAULT_MAX_STROKES_PER_SAMPLE)
    parser.add_argument("--stop-on-non-improvement", action="store_true")
    parser.add_argument("--min-pass-mad-improvement", type=float, default=DEFAULT_MIN_PASS_MAD_IMPROVEMENT)
    parser.add_argument("--keep-structure-failed-passes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allow-visual-failed-checkpoint", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
