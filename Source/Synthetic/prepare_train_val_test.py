"""Prepare BrushWright train/val/test folders from generated PT samples."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SOURCE_ROOT = Path("Outputs/Latest/PaintTransformerSamples/ArtInstituteChicago")
DEFAULT_OUTPUT_ROOT = Path("Data")
DEFAULT_BASE_COUNT = None
DEFAULT_FINISHING_COUNT = None
DEFAULT_COMPLETION_RATIO = 0.50
DEFAULT_MIN_COMPLETION = 0.45
DEFAULT_MAX_COMPLETION = 0.55
DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO = 3.0 / 5.0
DEFAULT_DRAFT_IMAGE_MIN_COMPLETION = 0.50
DEFAULT_DRAFT_IMAGE_MAX_COMPLETION = 0.70
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_SEED = 20260602
TARGET_SELECTION_ORDERED = "ordered"
TARGET_SELECTION_STRUCTURE_FIRST = "structure-first"
TARGET_SELECTION_STRUCTURE_FIRST_VERSION = "structure_first_v1"
TARGET_SELECTION_VISUAL_DELTA = "visual-delta"
TARGET_SELECTION_VISUAL_DELTA_VERSION = "visual_delta_v1"
TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET = "paint_transformer_original_image_target_v1"
TARGET_CONTRACT_OUTPUT_DETAIL_PAIR = "paint_transformer_output_detail_pair_v1"
TARGET_CONTRACT_LEGACY_RESPLIT = "paint_transformer_resplit_v1"
DRAFT_SOURCE_NATIVE_FRAME = "paint_transformer_native_frame"
DRAFT_SOURCE_STROKE_COUNT_RENDER = "paint_transformer_stroke_count_render"
DRAFT_SOURCE_OUTPUT_DRAFT = "paint_transformer_output_draft"
TARGET_SOURCE_ORIGINAL_IMAGE = "source_original_image"
TARGET_SOURCE_NATIVE_FINAL = "paint_transformer_native_final"
STRUCTURE_FIRST_COARSE_COUNT = 16


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Data/Train, Data/Val, and Data/Test datasets.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-count", type=int, default=DEFAULT_BASE_COUNT)
    parser.add_argument("--finishing-count", type=int, default=DEFAULT_FINISHING_COUNT)
    parser.add_argument("--completion-ratio", type=float, default=DEFAULT_COMPLETION_RATIO)
    parser.add_argument("--min-completion", type=float, default=DEFAULT_MIN_COMPLETION)
    parser.add_argument("--max-completion", type=float, default=DEFAULT_MAX_COMPLETION)
    parser.add_argument("--draft-image-completion-ratio", type=float, default=DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO)
    parser.add_argument("--draft-image-min-completion", type=float, default=DEFAULT_DRAFT_IMAGE_MIN_COMPLETION)
    parser.add_argument("--draft-image-max-completion", type=float, default=DEFAULT_DRAFT_IMAGE_MAX_COMPLETION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--target-selection-mode",
        choices=(TARGET_SELECTION_ORDERED, TARGET_SELECTION_STRUCTURE_FIRST, TARGET_SELECTION_VISUAL_DELTA),
        default=TARGET_SELECTION_ORDERED,
    )
    parser.add_argument("--render-draft-from-base", action="store_true")
    parser.add_argument(
        "--use-output-detail-pair",
        action="store_true",
        help="Use each PaintTransformer sample's own draft/final images and detail stroke split.",
    )
    parser.add_argument(
        "--use-50-stroke-count-full",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-native-50-full",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args(argv)

    prepare_splits(
        source_root=args.source_root,
        output_root=args.output_root,
        base_count=args.base_count,
        finishing_count=args.finishing_count,
        completion_ratio=args.completion_ratio,
        min_completion=args.min_completion,
        max_completion=args.max_completion,
        draft_image_completion_ratio=args.draft_image_completion_ratio,
        draft_image_min_completion=args.draft_image_min_completion,
        draft_image_max_completion=args.draft_image_max_completion,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        target_selection_mode=args.target_selection_mode,
        render_draft_from_base=args.render_draft_from_base,
        use_output_detail_pair=args.use_output_detail_pair or args.use_50_stroke_count_full or args.use_native_50_full,
        seed=args.seed,
        limit=args.limit,
        clear_existing=args.clear_existing,
    )
    return 0


def prepare_splits(
    source_root: Path,
    output_root: Path,
    base_count: int | None = DEFAULT_BASE_COUNT,
    finishing_count: int | None = DEFAULT_FINISHING_COUNT,
    completion_ratio: float = DEFAULT_COMPLETION_RATIO,
    min_completion: float = DEFAULT_MIN_COMPLETION,
    max_completion: float = DEFAULT_MAX_COMPLETION,
    draft_image_completion_ratio: float = DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
    draft_image_min_completion: float = DEFAULT_DRAFT_IMAGE_MIN_COMPLETION,
    draft_image_max_completion: float = DEFAULT_DRAFT_IMAGE_MAX_COMPLETION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    target_selection_mode: str = TARGET_SELECTION_ORDERED,
    render_draft_from_base: bool = False,
    use_output_detail_pair: bool = False,
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
    clear_existing: bool = False,
) -> dict[str, Any]:
    _validate_args(
        base_count,
        finishing_count,
        completion_ratio,
        min_completion,
        max_completion,
        draft_image_completion_ratio,
        draft_image_min_completion,
        draft_image_max_completion,
        val_fraction,
        test_fraction,
        target_selection_mode,
        render_draft_from_base,
        use_output_detail_pair,
    )
    target_selection_mode = _effective_target_selection_mode(
        target_selection_mode=target_selection_mode,
        use_output_detail_pair=use_output_detail_pair,
    )
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    sample_dirs = sorted(path.parent for path in source_root.glob("*/sample.json"))
    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    if not sample_dirs:
        raise ValueError(f"no samples found under {source_root}")

    rng = random.Random(seed)
    rng.shuffle(sample_dirs)
    split_dirs = _split_sample_dirs(sample_dirs, val_fraction=val_fraction, test_fraction=test_fraction)

    if clear_existing:
        for split_name in ("Train", "Val", "Test"):
            split_path = output_root / split_name
            if split_path.exists():
                _rmtree_with_retries(split_path)

    output_root.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    for split_name, dirs in split_dirs.items():
        split_root = output_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        written_samples = []
        for source_dir in dirs:
            destination_dir = split_root / source_dir.name
            if destination_dir.exists():
                raise FileExistsError(f"destination already exists: {destination_dir}")
            written_samples.append(
                _write_sample(
                    source_dir=source_dir,
                    destination_dir=destination_dir,
                    split_name=split_name,
                    requested_base_count=base_count,
                    requested_finishing_count=finishing_count,
                    completion_ratio=completion_ratio,
                    min_completion=min_completion,
                    max_completion=max_completion,
                    draft_image_completion_ratio=draft_image_completion_ratio,
                    draft_image_min_completion=draft_image_min_completion,
                    draft_image_max_completion=draft_image_max_completion,
                    target_selection_mode=target_selection_mode,
                    render_draft_from_base=render_draft_from_base,
                    use_output_detail_pair=use_output_detail_pair,
                )
            )
        split_manifest = {
            "version": 1,
            "split": split_name,
            "sample_count": len(written_samples),
            "base_count": base_count,
            "finishing_count": finishing_count,
            "completion_ratio": completion_ratio,
            "min_completion": min_completion,
            "max_completion": max_completion,
            "draft_image_completion_ratio": draft_image_completion_ratio,
            "draft_image_min_completion": draft_image_min_completion,
            "draft_image_max_completion": draft_image_max_completion,
            "target_selection_mode": _metadata_target_selection_mode(target_selection_mode),
            "render_draft_from_base": render_draft_from_base,
            "target_contract": _target_contract(use_output_detail_pair),
            "draft_source": _dataset_draft_source(
                use_output_detail_pair=use_output_detail_pair,
                render_draft_from_base=render_draft_from_base,
            ),
            "target_source": TARGET_SOURCE_ORIGINAL_IMAGE,
            "samples": written_samples,
        }
        _write_json(split_root / "dataset_manifest.json", split_manifest)
        split_summaries[split_name] = {
            "sample_count": len(written_samples),
            "manifest": str((split_root / "dataset_manifest.json").relative_to(output_root)),
        }

    dataset_manifest = {
        "version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "completion_ratio": completion_ratio,
        "min_completion": min_completion,
        "max_completion": max_completion,
        "draft_image_completion_ratio": draft_image_completion_ratio,
        "draft_image_min_completion": draft_image_min_completion,
        "draft_image_max_completion": draft_image_max_completion,
        "target_selection_mode": _metadata_target_selection_mode(target_selection_mode),
        "render_draft_from_base": render_draft_from_base,
        "target_contract": _target_contract(use_output_detail_pair),
        "draft_source": _dataset_draft_source(
            use_output_detail_pair=use_output_detail_pair,
            render_draft_from_base=render_draft_from_base,
        ),
        "target_source": TARGET_SOURCE_ORIGINAL_IMAGE,
        "total_samples": len(sample_dirs),
        "splits": split_summaries,
    }
    _write_json(output_root / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest


def _write_sample(
    source_dir: Path,
    destination_dir: Path,
    split_name: str,
    requested_base_count: int | None,
    requested_finishing_count: int | None,
    completion_ratio: float,
    min_completion: float,
    max_completion: float,
    draft_image_completion_ratio: float,
    draft_image_min_completion: float,
    draft_image_max_completion: float,
    target_selection_mode: str,
    render_draft_from_base: bool,
    use_output_detail_pair: bool,
) -> dict[str, Any]:
    source_sample = _read_json(source_dir / "sample.json")
    full_program = _read_json(source_dir / source_sample["full_program"])
    resolved_base_count, resolved_finishing_count = _resolve_requested_counts(
        source_sample=source_sample,
        full_program=full_program,
        requested_base_count=requested_base_count,
        requested_finishing_count=requested_finishing_count,
        use_output_detail_pair=use_output_detail_pair,
    )
    strokes = _select_source_strokes_for_request(
        source_dir=source_dir,
        source_sample=source_sample,
        full_program=full_program,
        requested_base_count=resolved_base_count,
        requested_finishing_count=resolved_finishing_count,
    )
    desired_stroke_count = None
    if use_output_detail_pair:
        desired_stroke_count = len(full_program["strokes"])
        strokes = _available_strokes_or_default(source_dir=source_dir, source_sample=source_sample, default_strokes=strokes)
    total_count = len(strokes)
    render_manifest = _read_json(source_dir / source_sample["finished_render_manifest"])
    finished_image_path = source_dir / source_sample["finished_image"]
    original_target_image_path = _source_image_path(source_dir=source_dir, source_sample=source_sample)
    if use_output_detail_pair:
        draft_frame_path, actual_draft_image_completion_ratio, native_frame_index = _select_draft_frame(
            render_manifest=render_manifest,
            completion_ratio=draft_image_completion_ratio,
            min_completion=draft_image_min_completion,
            max_completion=draft_image_max_completion,
        )
        native_frame_count = int(render_manifest["native_frame_count"])
        split_completion_ratio = actual_draft_image_completion_ratio
        split_min_completion = draft_image_min_completion
        split_max_completion = draft_image_max_completion
    elif requested_base_count is None and requested_finishing_count is None:
        draft_frame_path, actual_draft_image_completion_ratio, native_frame_index = _select_draft_frame(
            render_manifest=render_manifest,
            completion_ratio=draft_image_completion_ratio,
            min_completion=draft_image_min_completion,
            max_completion=draft_image_max_completion,
        )
        split_completion_ratio = actual_draft_image_completion_ratio
        split_min_completion = draft_image_min_completion
        split_max_completion = draft_image_max_completion
    else:
        split_completion_ratio = completion_ratio
        split_min_completion = 0.0
        split_max_completion = 1.0

    selected_strokes, base_count, finishing_count, adjusted_counts, target_selection_manifest = _select_dataset_strokes(
        source_dir=source_dir,
        strokes=strokes,
        requested_base_count=resolved_base_count,
        requested_finishing_count=resolved_finishing_count,
        completion_ratio=split_completion_ratio,
        target_selection_mode=target_selection_mode,
        draft_image_path=draft_frame_path if use_output_detail_pair else None,
        target_image_path=original_target_image_path if use_output_detail_pair else None,
        desired_stroke_count=desired_stroke_count,
    )
    total_count = len(selected_strokes)
    actual_completion_ratio = base_count / total_count
    if not adjusted_counts and (actual_completion_ratio < split_min_completion or actual_completion_ratio > split_max_completion):
        raise ValueError(f"stroke completion ratio out of range for {source_dir}: {actual_completion_ratio:.3f}")

    if (
        not use_output_detail_pair
        and resolved_base_count is not None
        and resolved_finishing_count is not None
        and not render_draft_from_base
    ):
        draft_frame_path, actual_draft_image_completion_ratio, native_frame_index = _select_draft_frame(
            render_manifest=render_manifest,
            completion_ratio=actual_completion_ratio,
            min_completion=0.0,
            max_completion=1.0,
        )
    if (
        not use_output_detail_pair
        and resolved_base_count is not None
        and resolved_finishing_count is not None
        and render_draft_from_base
    ):
        actual_draft_image_completion_ratio = actual_completion_ratio
        native_frame_index = base_count - 1
    target_selection_mode_metadata = _metadata_target_selection_mode(target_selection_mode)
    target_contract = _target_contract(use_output_detail_pair)
    draft_source = _dataset_draft_source(
        use_output_detail_pair=use_output_detail_pair,
        render_draft_from_base=render_draft_from_base,
    )
    target_source = TARGET_SOURCE_ORIGINAL_IMAGE

    metadata = dict(full_program.get("metadata", {}))
    metadata.update(
        {
            "dataset_split": split_name,
            "dataset_source_sample": str(source_dir),
            "completion_ratio": actual_completion_ratio,
            "draft_image_completion_ratio": actual_draft_image_completion_ratio,
            "draft_stroke_completion_delta": abs(actual_draft_image_completion_ratio - actual_completion_ratio),
            "base_count": base_count,
            "finishing_count": finishing_count,
            "stroke_count_adjusted": adjusted_counts,
            "target_selection_mode": target_selection_mode_metadata,
            "render_draft_from_base": render_draft_from_base,
            "target_contract": target_contract,
            "draft_source": draft_source,
            "target_source": target_source,
            "actual_draft_ratio": actual_draft_image_completion_ratio,
        }
    )

    destination_dir.mkdir(parents=True, exist_ok=True)
    base_program = _program(
        full_program,
        metadata | {"split": "base", "stroke_count": base_count},
        selected_strokes[:base_count],
    )
    finishing_program = _program(
        full_program,
        metadata | {"split": "finishing", "stroke_count": finishing_count},
        selected_strokes[base_count:],
    )
    resplit_full_program = _program(full_program, metadata | {"stroke_count": total_count}, selected_strokes)

    _write_json(destination_dir / "full_program.json", resplit_full_program)
    _write_json(destination_dir / "base_strokes.json", base_program)
    _write_json(destination_dir / "finishing_strokes.json", finishing_program)
    if target_selection_manifest is not None:
        _write_json(destination_dir / "target_selection_manifest.json", target_selection_manifest)

    if use_output_detail_pair:
        _link_or_copy(draft_frame_path, destination_dir / "draft.png")
        _link_or_copy(finished_image_path, destination_dir / "finished.png")
        _assert_same_file_contents(destination_dir / "draft.png", draft_frame_path, "source PaintTransformer draft")
        _assert_same_file_contents(destination_dir / "finished.png", finished_image_path, "source PaintTransformer final")
        _write_render_manifest(
            destination_dir=destination_dir,
            render_dir_name="draft_render",
            image_name="draft.png",
            source_image=draft_frame_path,
            native_frame_index=native_frame_index,
            native_frame_count=native_frame_count,
            image_completion_ratio=actual_draft_image_completion_ratio,
        )
        _write_render_manifest(
            destination_dir=destination_dir,
            render_dir_name="finished_render",
            image_name="finished.png",
            source_image=finished_image_path,
            native_frame_index=render_manifest["native_frame_count"] - 1,
            native_frame_count=render_manifest["native_frame_count"],
            image_completion_ratio=1.0,
        )
    elif render_draft_from_base:
        render_strokes, render_base_count, render_context_count = _render_context_strokes(
            source_dir=source_dir,
            source_sample=source_sample,
            full_program=full_program,
            selected_strokes=selected_strokes,
            base_count=base_count,
        )
        render_context_program = _program(
            full_program,
            metadata
            | {
                "split": "render_context",
                "stroke_count": len(render_strokes),
                "render_context_count": render_context_count,
                "render_context_base_count": render_base_count,
                "render_context_target_count": len(render_strokes),
            },
            render_strokes,
        )
        _write_json(destination_dir / "render_context_program.json", render_context_program)
        _render_split_program_images(
            full_program_path=destination_dir / "render_context_program.json",
            base_count=render_base_count,
            draft_render_dir=destination_dir / "draft_render",
            draft_image_path=destination_dir / "draft.png",
            finished_render_dir=destination_dir / "finished_render",
            finished_image_path=destination_dir / "finished.png",
        )
        draft_frame_path = destination_dir / "draft_render" / "final.png"
        finished_image_path = destination_dir / "finished_render" / "final.png"
    else:
        _link_or_copy(draft_frame_path, destination_dir / "draft.png")
        _link_or_copy(finished_image_path, destination_dir / "finished.png")

        _write_render_manifest(
            destination_dir=destination_dir,
            render_dir_name="draft_render",
            image_name="draft.png",
            source_image=draft_frame_path,
            native_frame_index=native_frame_index,
            native_frame_count=render_manifest["native_frame_count"],
            image_completion_ratio=actual_draft_image_completion_ratio,
        )
        _write_render_manifest(
            destination_dir=destination_dir,
            render_dir_name="finished_render",
            image_name="finished.png",
            source_image=finished_image_path,
            native_frame_index=render_manifest["native_frame_count"] - 1,
            native_frame_count=render_manifest["native_frame_count"],
            image_completion_ratio=1.0,
        )
    _write_resized_target_image(
        source_image=original_target_image_path,
        draft_image=destination_dir / "draft.png",
        output_path=destination_dir / "target.png",
    )
    split_manifest = {
        "version": 1,
        "method": "dataset_resplit",
        "source_sample": str(source_dir),
        "total_strokes": total_count,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "requested_base_count": resolved_base_count,
        "requested_finishing_count": resolved_finishing_count,
        "stroke_count_adjusted": adjusted_counts,
        "completion_ratio": actual_completion_ratio,
        "draft_image_completion_ratio": actual_draft_image_completion_ratio,
        "draft_stroke_completion_delta": abs(actual_draft_image_completion_ratio - actual_completion_ratio),
        "native_frame_index": native_frame_index,
        "native_frame_count": render_manifest["native_frame_count"],
        "withheld_start_index": base_count,
        "withheld_end_index_exclusive": total_count,
        "target_selection_mode": target_selection_mode_metadata,
        "render_draft_from_base": render_draft_from_base,
        "target_contract": target_contract,
        "draft_source": draft_source,
        "target_source": target_source,
        "actual_draft_ratio": actual_draft_image_completion_ratio,
        "render_context_program": "render_context_program.json" if render_draft_from_base and not use_output_detail_pair else None,
        "render_context_count": render_context_count if render_draft_from_base and not use_output_detail_pair else 0,
        "render_context_base_count": render_base_count if render_draft_from_base and not use_output_detail_pair else base_count,
    }
    _write_json(destination_dir / "split_manifest.json", split_manifest)

    sample = {
        "version": 1,
        "sample_id": source_sample["sample_id"],
        "dataset_split": split_name,
        "source_sample": str(source_dir),
        "source_image": source_sample["source_image"],
        "canvas": full_program["canvas"],
        "stroke_count": total_count,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "requested_base_count": resolved_base_count,
        "requested_finishing_count": resolved_finishing_count,
        "stroke_count_adjusted": adjusted_counts,
        "completion_ratio": actual_completion_ratio,
        "draft_image_completion_ratio": actual_draft_image_completion_ratio,
        "draft_stroke_completion_delta": abs(actual_draft_image_completion_ratio - actual_completion_ratio),
        "target_selection_mode": target_selection_mode_metadata,
        "target_selection_manifest": "target_selection_manifest.json" if target_selection_manifest is not None else None,
        "render_draft_from_base": render_draft_from_base,
        "target_contract": target_contract,
        "draft_source": draft_source,
        "target_source": target_source,
        "actual_draft_ratio": actual_draft_image_completion_ratio,
        "generator": "paint_transformer_reference_resplit",
        "full_program": "full_program.json",
        "base_strokes": "base_strokes.json",
        "finishing_strokes": "finishing_strokes.json",
        "draft_image": "draft.png",
        "target_image": "target.png",
        "finished_image": "finished.png",
        "draft_render_manifest": "draft_render/render_manifest.json",
        "finished_render_manifest": "finished_render/render_manifest.json",
        "split_manifest": "split_manifest.json",
    }
    if render_draft_from_base and not use_output_detail_pair:
        sample["render_context_program"] = "render_context_program.json"
        sample["render_context_count"] = render_context_count
        sample["render_context_base_count"] = render_base_count
    _write_json(destination_dir / "sample.json", sample)
    return {
        "sample_id": sample["sample_id"],
        "path": destination_dir.name,
        "completion_ratio": actual_completion_ratio,
        "draft_image_completion_ratio": actual_draft_image_completion_ratio,
        "draft_stroke_completion_delta": abs(actual_draft_image_completion_ratio - actual_completion_ratio),
        "stroke_count_adjusted": adjusted_counts,
        "target_selection_mode": target_selection_mode_metadata,
        "target_contract": target_contract,
        "draft_source": draft_source,
        "target_source": target_source,
    }


def _resolve_requested_counts(
    source_sample: dict[str, Any],
    full_program: dict[str, Any],
    requested_base_count: int | None,
    requested_finishing_count: int | None,
    use_output_detail_pair: bool,
) -> tuple[int | None, int | None]:
    if requested_base_count is not None or requested_finishing_count is not None:
        return requested_base_count, requested_finishing_count
    if not use_output_detail_pair:
        return None, None
    return None, None


def _select_draft_frame(
    render_manifest: dict[str, Any],
    completion_ratio: float,
    min_completion: float,
    max_completion: float,
) -> tuple[Path, float, int]:
    native_frame_count = render_manifest["native_frame_count"]
    candidates = []
    for index, frame_path in enumerate(render_manifest["frames"]):
        frame_completion = (index + 1) / native_frame_count
        if min_completion <= frame_completion <= max_completion:
            candidates.append((abs(frame_completion - completion_ratio), index, frame_completion, Path(frame_path)))
    if not candidates:
        for index, frame_path in enumerate(render_manifest["frames"]):
            frame_completion = (index + 1) / native_frame_count
            candidates.append((abs(frame_completion - completion_ratio), index, frame_completion, Path(frame_path)))
    _, index, frame_completion, frame_path = min(candidates)
    if not frame_path.exists():
        raise OSError(f"missing native frame: {frame_path}")
    return frame_path, frame_completion, index


def _select_source_strokes_for_request(
    source_dir: Path,
    source_sample: dict[str, Any],
    full_program: dict[str, Any],
    requested_base_count: int | None,
    requested_finishing_count: int | None,
) -> list[dict[str, Any]]:
    full_strokes = full_program["strokes"]
    if requested_base_count is None or requested_finishing_count is None:
        return full_strokes

    requested_count = requested_base_count + requested_finishing_count
    if len(full_strokes) >= requested_count:
        return full_strokes

    available_path = source_sample.get("available_strokes", "available_strokes.json")
    available_program_path = source_dir / available_path
    if not available_program_path.exists():
        return full_strokes

    available_program = _read_json(available_program_path)
    available_strokes = available_program["strokes"]
    if len(available_strokes) <= len(full_strokes):
        return full_strokes

    stroke_window = full_program.get("metadata", {}).get("stroke_window", "detail")
    if stroke_window == "detail" and len(available_strokes) > requested_count:
        return available_strokes[-requested_count:]
    return available_strokes


def _available_strokes_or_default(
    source_dir: Path,
    source_sample: dict[str, Any],
    default_strokes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available_path = source_sample.get("available_strokes", "available_strokes.json")
    available_program_path = source_dir / available_path
    if not available_program_path.exists():
        return default_strokes
    available_program = _read_json(available_program_path)
    available_strokes = available_program.get("strokes", [])
    if len(available_strokes) <= len(default_strokes):
        return default_strokes
    return available_strokes


def _render_context_strokes(
    source_dir: Path,
    source_sample: dict[str, Any],
    full_program: dict[str, Any],
    selected_strokes: list[dict[str, Any]],
    base_count: int,
) -> tuple[list[dict[str, Any]], int, int]:
    metadata = full_program.get("metadata", {})
    selected_start_index = int(metadata.get("selected_start_index", 0) or 0)
    if selected_start_index <= 0:
        return selected_strokes, base_count, 0

    available_path = source_sample.get("available_strokes", "available_strokes.json")
    available_program_path = source_dir / available_path
    if not available_program_path.exists():
        return selected_strokes, base_count, 0

    available_program = _read_json(available_program_path)
    available_strokes = available_program["strokes"]
    render_total_count = selected_start_index + len(selected_strokes)
    if render_total_count > len(available_strokes):
        return selected_strokes, base_count, 0

    render_base_count = selected_start_index + base_count
    return available_strokes[:render_total_count], render_base_count, selected_start_index


def _select_dataset_strokes(
    source_dir: Path,
    strokes: list[dict[str, Any]],
    requested_base_count: int | None,
    requested_finishing_count: int | None,
    completion_ratio: float,
    target_selection_mode: str,
    draft_image_path: Path | None = None,
    target_image_path: Path | None = None,
    desired_stroke_count: int | None = None,
) -> tuple[list[dict[str, Any]], int, int, bool, dict[str, Any] | None]:
    total_count = len(strokes)
    split_total_count = min(total_count, desired_stroke_count) if desired_stroke_count is not None else total_count
    if requested_base_count is None and requested_finishing_count is None:
        base_count = max(1, min(split_total_count - 1, round(split_total_count * completion_ratio)))
        finishing_count = split_total_count - base_count
        if target_selection_mode == TARGET_SELECTION_VISUAL_DELTA:
            if draft_image_path is None or target_image_path is None:
                raise ValueError("visual-delta target selection requires draft and target images")
            return _select_visual_delta_strokes(
                source_dir=source_dir,
                strokes=strokes,
                base_count=base_count,
                finishing_count=finishing_count,
                draft_image_path=draft_image_path,
                target_image_path=target_image_path,
            )
        return strokes[:split_total_count], base_count, finishing_count, split_total_count != total_count, None
    if requested_base_count is None or requested_finishing_count is None:
        raise ValueError("--base-count and --finishing-count must be provided together")
    if requested_base_count <= 0 or requested_finishing_count <= 0:
        raise ValueError("base_count and finishing_count must be positive")

    required_count = requested_base_count + requested_finishing_count
    if target_selection_mode == TARGET_SELECTION_STRUCTURE_FIRST and total_count > requested_base_count:
        return _select_structure_first_strokes(
            source_dir=source_dir,
            strokes=strokes,
            requested_base_count=requested_base_count,
            requested_finishing_count=requested_finishing_count,
            completion_ratio=completion_ratio,
        )
    if target_selection_mode == TARGET_SELECTION_VISUAL_DELTA:
        if draft_image_path is None or target_image_path is None:
            raise ValueError("visual-delta target selection requires draft and target images")
        return _select_visual_delta_strokes(
            source_dir=source_dir,
            strokes=strokes,
            base_count=requested_base_count,
            finishing_count=requested_finishing_count,
            draft_image_path=draft_image_path,
            target_image_path=target_image_path,
        )
    if total_count >= required_count:
        return strokes[:required_count], requested_base_count, requested_finishing_count, False, None
    if total_count > requested_base_count:
        return strokes, requested_base_count, total_count - requested_base_count, True, None
    if total_count < 2:
        raise ValueError(f"{source_dir} has only {total_count} stroke; need at least 2")

    base_count = max(1, min(total_count - 1, round(total_count * completion_ratio)))
    finishing_count = total_count - base_count
    return strokes, base_count, finishing_count, True, None


def _select_visual_delta_strokes(
    source_dir: Path,
    strokes: list[dict[str, Any]],
    base_count: int,
    finishing_count: int,
    draft_image_path: Path,
    target_image_path: Path,
) -> tuple[list[dict[str, Any]], int, int, bool, dict[str, Any] | None]:
    if len(strokes) < 2:
        raise ValueError(f"{source_dir} has only {len(strokes)} stroke; need at least 2")
    selected_count = min(max(1, finishing_count), len(strokes) - 1)
    scored = _score_visual_delta_candidates(strokes, draft_image_path=draft_image_path, target_image_path=target_image_path)
    selected_entries = _select_visual_delta_candidates(scored, count=selected_count)
    selected_indices = {int(entry["source_index"]) for entry in selected_entries}
    base_strokes = [stroke for index, stroke in enumerate(strokes) if index not in selected_indices]
    selected_base_count = min(max(1, base_count), len(base_strokes))
    selected_finishing = sorted(selected_entries, key=lambda entry: int(entry["source_index"]))
    selected_strokes = base_strokes[:selected_base_count] + [entry["stroke"] for entry in selected_finishing]
    adjusted_counts = selected_base_count != base_count or len(selected_finishing) != finishing_count
    manifest = _visual_delta_target_selection_manifest(
        source_dir=source_dir,
        base_count=selected_base_count,
        finishing_count=len(selected_finishing),
        adjusted_counts=adjusted_counts,
        selected_entries=selected_finishing,
        base_source_indices=[
            index for index in range(len(strokes)) if index not in selected_indices
        ][:selected_base_count],
    )
    return selected_strokes, selected_base_count, len(selected_finishing), adjusted_counts, manifest


def _score_visual_delta_candidates(
    strokes: list[dict[str, Any]],
    draft_image_path: Path,
    target_image_path: Path,
) -> list[dict[str, Any]]:
    error_map = _visual_delta_error_map(draft_image_path=draft_image_path, target_image_path=target_image_path)
    edge_map = _visual_delta_edge_map(error_map)
    scored = []
    for index, stroke in enumerate(strokes):
        score_terms = _score_stroke_against_visual_delta(stroke, error_map=error_map, edge_map=edge_map)
        scored.append(
            {
                "source_index": index,
                "stroke": stroke,
                **score_terms,
            }
        )
    scored.sort(key=lambda entry: (entry["score"], entry["local_error_mean"], entry["local_edge_mean"], entry["source_index"]), reverse=True)
    return scored


def _select_visual_delta_candidates(scored: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_points: list[tuple[float, float]] = []
    remaining = list(scored)
    while remaining and len(selected) < count:
        best_index = 0
        best_rank = -float("inf")
        for index, entry in enumerate(remaining):
            stroke = entry["stroke"]
            point = (float(stroke["x"]), float(stroke["y"]))
            diversity = _nearest_distance(point, selected_points) if selected_points else 1.0
            rank = float(entry["score"]) + diversity * 0.02
            if rank > best_rank:
                best_rank = rank
                best_index = index
        entry = remaining.pop(best_index)
        selected.append(entry)
        selected_points.append((float(entry["stroke"]["x"]), float(entry["stroke"]["y"])))
    return selected


def _score_stroke_against_visual_delta(
    stroke: dict[str, Any],
    error_map: Any,
    edge_map: Any,
) -> dict[str, float]:
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
    local_error = error_map[top:bottom, left:right]
    local_edge = edge_map[top:bottom, left:right]
    local_error_mean = float(local_error.mean()) if local_error.size else 0.0
    local_error_max = float(local_error.max()) if local_error.size else 0.0
    local_edge_mean = float(local_edge.mean()) if local_edge.size else 0.0
    center_error = float(error_map[y, x])
    area = max(1.0, length_pixels * width_pixels)
    area_score = min(8.0, math.sqrt(area))
    score = (
        0.45 * local_error_mean
        + 0.25 * center_error
        + 0.20 * local_error_max
        + 0.10 * local_edge_mean
    ) * area_score
    return {
        "score": score,
        "local_error_mean": local_error_mean,
        "local_error_max": local_error_max,
        "local_edge_mean": local_edge_mean,
        "center_error": center_error,
        "area_score": area_score,
    }


def _visual_delta_error_map(draft_image_path: Path, target_image_path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(draft_image_path) as draft_image, Image.open(target_image_path) as target_image:
        draft_rgb = draft_image.convert("RGB")
        target_rgb = target_image.convert("RGB")
        if target_rgb.size != draft_rgb.size:
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            target_rgb = target_rgb.resize(draft_rgb.size, resampling)
        draft = np.asarray(draft_rgb, dtype=np.float32) / 255.0
        target = np.asarray(target_rgb, dtype=np.float32) / 255.0
    return np.max(np.abs(target - draft), axis=2)


def _visual_delta_edge_map(error_map: Any) -> Any:
    import numpy as np

    horizontal = np.zeros_like(error_map)
    vertical = np.zeros_like(error_map)
    horizontal[:, 1:] = np.abs(error_map[:, 1:] - error_map[:, :-1])
    vertical[1:, :] = np.abs(error_map[1:, :] - error_map[:-1, :])
    return np.clip(horizontal + vertical, 0.0, 1.0)


def _select_structure_first_strokes(
    source_dir: Path,
    strokes: list[dict[str, Any]],
    requested_base_count: int,
    requested_finishing_count: int,
    completion_ratio: float,
) -> tuple[list[dict[str, Any]], int, int, bool, dict[str, Any] | None]:
    if len(strokes) < 2:
        raise ValueError(f"{source_dir} has only {len(strokes)} stroke; need at least 2")
    if len(strokes) <= requested_base_count:
        base_count = max(1, min(len(strokes) - 1, round(len(strokes) * completion_ratio)))
        manifest = _structure_target_selection_manifest(
            source_dir=source_dir,
            base_count=base_count,
            finishing_count=len(strokes) - base_count,
            adjusted_counts=True,
            selected_entries=[
                {
                    "source_index": index,
                    "stroke": stroke,
                    "score": 0.0,
                    "coverage_score": 0.0,
                    "color_delta": 0.0,
                    "texture_penalty": 0.0,
                }
                for index, stroke in enumerate(strokes[base_count:], start=base_count)
            ],
        )
        return strokes, base_count, len(strokes) - base_count, True, manifest

    base_strokes = strokes[:requested_base_count]
    candidate_strokes = strokes[requested_base_count:]
    selected_count = min(requested_finishing_count, len(candidate_strokes))
    coarse_count = min(STRUCTURE_FIRST_COARSE_COUNT, selected_count)
    detail_count = selected_count - coarse_count

    scored = _score_structure_candidates(candidate_strokes, base_strokes, requested_base_count)
    coarse = _select_diverse_candidates(scored, count=coarse_count, selected=[])
    detail = _select_diverse_candidates(
        [entry for entry in scored if entry not in coarse],
        count=detail_count,
        selected=coarse,
    )
    selected_entries = coarse + detail
    selected_strokes = base_strokes + [entry["stroke"] for entry in selected_entries]
    manifest = _structure_target_selection_manifest(
        source_dir=source_dir,
        base_count=requested_base_count,
        finishing_count=selected_count,
        adjusted_counts=selected_count != requested_finishing_count,
        selected_entries=selected_entries,
    )
    return selected_strokes, requested_base_count, selected_count, selected_count != requested_finishing_count, manifest


def _score_structure_candidates(
    candidate_strokes: list[dict[str, Any]],
    base_strokes: list[dict[str, Any]],
    first_candidate_index: int,
) -> list[dict[str, Any]]:
    base_color = _mean_stroke_color(base_strokes)
    scored = []
    for offset, stroke in enumerate(candidate_strokes):
        length = _stroke_float(stroke, "length", 0.0)
        width = _stroke_float(stroke, "width", 0.0)
        opacity = _stroke_float(stroke, "opacity", 1.0)
        color = stroke.get("color", [0.0, 0.0, 0.0])
        color_delta = sum(abs(float(color[index]) - base_color[index]) for index in range(3)) / 3.0
        area = length * width
        aspect = length / max(width, 1e-6)
        aspect_bonus = min(aspect, 8.0) / 8.0
        coverage_score = area * opacity
        structure_score = coverage_score * 1200.0 + color_delta * 3.0 + aspect_bonus * 0.75
        texture_penalty = 0.0
        if area < 0.00012:
            texture_penalty += 2.0
        if length < 0.008 or width < 0.004:
            texture_penalty += 1.0
        score = structure_score - texture_penalty
        scored.append(
            {
                "source_index": first_candidate_index + offset,
                "stroke": stroke,
                "score": score,
                "coverage_score": coverage_score,
                "color_delta": color_delta,
                "texture_penalty": texture_penalty,
            }
        )
    scored.sort(key=lambda entry: (entry["score"], entry["coverage_score"], -entry["source_index"]), reverse=True)
    return scored


def _select_diverse_candidates(
    scored: list[dict[str, Any]],
    count: int,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    selected_points = [(float(entry["stroke"]["x"]), float(entry["stroke"]["y"])) for entry in selected]
    remaining = list(scored)
    while remaining and len(chosen) < count:
        best_index = 0
        best_rank = -float("inf")
        for index, entry in enumerate(remaining):
            x = float(entry["stroke"]["x"])
            y = float(entry["stroke"]["y"])
            diversity = _nearest_distance((x, y), selected_points) if selected_points else 1.0
            rank = float(entry["score"]) + diversity * 0.5
            if rank > best_rank:
                best_rank = rank
                best_index = index
        entry = remaining.pop(best_index)
        chosen.append(entry)
        selected_points.append((float(entry["stroke"]["x"]), float(entry["stroke"]["y"])))
    return chosen


def _structure_target_selection_manifest(
    source_dir: Path,
    base_count: int,
    finishing_count: int,
    adjusted_counts: bool,
    selected_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = []
    for offset, entry in enumerate(selected_entries):
        stroke = entry["stroke"]
        selected.append(
            {
                "rank": offset,
                "source_index": int(entry["source_index"]),
                "x": float(stroke["x"]),
                "y": float(stroke["y"]),
                "score": float(entry.get("score", 0.0)),
                "coverage_score": float(entry.get("coverage_score", 0.0)),
                "color_delta": float(entry.get("color_delta", 0.0)),
                "texture_penalty": float(entry.get("texture_penalty", 0.0)),
                "role": "coarse" if offset < min(STRUCTURE_FIRST_COARSE_COUNT, finishing_count) else "detail",
            }
        )
    return {
        "version": 1,
        "target_selection_mode": TARGET_SELECTION_STRUCTURE_FIRST_VERSION,
        "source_sample": str(source_dir),
        "base_count": base_count,
        "finishing_count": finishing_count,
        "stroke_count_adjusted": adjusted_counts,
        "coarse_count": min(STRUCTURE_FIRST_COARSE_COUNT, finishing_count),
        "detail_count": max(0, finishing_count - min(STRUCTURE_FIRST_COARSE_COUNT, finishing_count)),
        "selected_source_indices": [entry["source_index"] for entry in selected],
        "selected": selected,
        "score_terms": {
            "coverage": "length * width * opacity",
            "structure_proxy": "coverage + color contrast + aspect bonus",
            "texture_penalty": "tiny-area and tiny-size penalties",
        },
    }


def _visual_delta_target_selection_manifest(
    source_dir: Path,
    base_count: int,
    finishing_count: int,
    adjusted_counts: bool,
    selected_entries: list[dict[str, Any]],
    base_source_indices: list[int],
) -> dict[str, Any]:
    selected = []
    for rank, entry in enumerate(selected_entries):
        stroke = entry["stroke"]
        selected.append(
            {
                "rank": rank,
                "source_index": int(entry["source_index"]),
                "x": float(stroke["x"]),
                "y": float(stroke["y"]),
                "score": float(entry.get("score", 0.0)),
                "local_error_mean": float(entry.get("local_error_mean", 0.0)),
                "local_error_max": float(entry.get("local_error_max", 0.0)),
                "local_edge_mean": float(entry.get("local_edge_mean", 0.0)),
                "center_error": float(entry.get("center_error", 0.0)),
                "area_score": float(entry.get("area_score", 0.0)),
                "role": "visual_delta",
            }
        )
    return {
        "version": 1,
        "target_selection_mode": TARGET_SELECTION_VISUAL_DELTA_VERSION,
        "source_sample": str(source_dir),
        "base_count": base_count,
        "finishing_count": finishing_count,
        "stroke_count_adjusted": adjusted_counts,
        "base_source_indices": base_source_indices,
        "selected_source_indices": [entry["source_index"] for entry in selected],
        "selected": selected,
        "score_terms": {
            "local_error_mean": "mean draft-target max-channel error near stroke center",
            "center_error": "draft-target max-channel error at stroke center",
            "local_error_max": "max draft-target error near stroke center",
            "local_edge_mean": "mean edge energy in the local error map",
            "area_score": "bounded sqrt(length_pixels * width_pixels)",
        },
    }


def _mean_stroke_color(strokes: list[dict[str, Any]]) -> list[float]:
    if not strokes:
        return [0.0, 0.0, 0.0]
    totals = [0.0, 0.0, 0.0]
    for stroke in strokes:
        color = stroke.get("color", [0.0, 0.0, 0.0])
        for index in range(3):
            totals[index] += float(color[index])
    return [value / len(strokes) for value in totals]


def _nearest_distance(point: tuple[float, float], selected_points: list[tuple[float, float]]) -> float:
    if not selected_points:
        return 1.0
    return min(math.dist(point, selected) for selected in selected_points)


def _stroke_float(stroke: dict[str, Any], field: str, default: float) -> float:
    try:
        return float(stroke.get(field, default))
    except (TypeError, ValueError):
        return default


def _metadata_target_selection_mode(target_selection_mode: str) -> str:
    if target_selection_mode == TARGET_SELECTION_STRUCTURE_FIRST:
        return TARGET_SELECTION_STRUCTURE_FIRST_VERSION
    if target_selection_mode == TARGET_SELECTION_VISUAL_DELTA:
        return TARGET_SELECTION_VISUAL_DELTA_VERSION
    return TARGET_SELECTION_ORDERED


def _effective_target_selection_mode(target_selection_mode: str, use_output_detail_pair: bool) -> str:
    if use_output_detail_pair and target_selection_mode == TARGET_SELECTION_ORDERED:
        return TARGET_SELECTION_VISUAL_DELTA
    return target_selection_mode


def _target_contract(use_output_detail_pair: bool) -> str:
    return TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET


def _draft_source(render_draft_from_base: bool) -> str:
    if render_draft_from_base:
        return DRAFT_SOURCE_STROKE_COUNT_RENDER
    return DRAFT_SOURCE_NATIVE_FRAME


def _dataset_draft_source(use_output_detail_pair: bool, render_draft_from_base: bool) -> str:
    if use_output_detail_pair:
        return DRAFT_SOURCE_NATIVE_FRAME
    return _draft_source(render_draft_from_base)


def _target_source(render_draft_from_base: bool) -> str:
    if render_draft_from_base:
        return "paint_transformer_stroke_count_render"
    return TARGET_SOURCE_NATIVE_FINAL


def _source_image_path(source_dir: Path, source_sample: dict[str, Any]) -> Path:
    source_image = source_sample.get("source_image")
    if not source_image:
        raise ValueError(f"{source_dir} sample is missing source_image metadata")
    path = Path(str(source_image)).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    if not path.exists():
        raise ValueError(f"source image does not exist: {path}")
    return path


def _write_resized_target_image(source_image: Path, draft_image: Path, output_path: Path) -> None:
    from PIL import Image

    with Image.open(draft_image) as draft, Image.open(source_image) as source:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        target = source.convert("RGB").resize(draft.size, resampling)
        target.save(output_path)


def _write_render_manifest(
    destination_dir: Path,
    render_dir_name: str,
    image_name: str,
    source_image: Path,
    native_frame_index: int,
    native_frame_count: int,
    image_completion_ratio: float,
) -> None:
    render_dir = destination_dir / render_dir_name
    render_dir.mkdir(parents=True, exist_ok=True)
    final_path = render_dir / "final.png"
    _link_or_copy(destination_dir / image_name, final_path)
    _write_json(
        render_dir / "render_manifest.json",
        {
            "version": 1,
            "renderer": "paint_transformer_native_inference_resplit",
            "source_image": str(source_image),
            "native_frame_index": native_frame_index,
            "native_frame_count": native_frame_count,
            "image_completion_ratio": image_completion_ratio,
            "final_image": str(final_path),
        },
    )


def _render_program_to_sample_image(program_path: Path, render_dir: Path, image_path: Path) -> None:
    from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer

    render_dir.mkdir(parents=True, exist_ok=True)
    render_program_final_with_paint_transformer(program_path, render_dir / "final.png")
    _link_or_copy(render_dir / "final.png", image_path)
    program = _read_json(program_path)
    _write_json(
        render_dir / "render_manifest.json",
        {
            "version": 1,
            "renderer": "paint_transformer_reference_final_only",
            "input": str(program_path),
            "canvas": program["canvas"],
            "stroke_count": len(program["strokes"]),
            "frame_count": 1,
            "final_image": str(render_dir / "final.png"),
        },
    )


def _render_split_program_images(
    full_program_path: Path,
    base_count: int,
    draft_render_dir: Path,
    draft_image_path: Path,
    finished_render_dir: Path,
    finished_image_path: Path,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Paint Transformer split rendering requires PyTorch") from exc

    from PIL import Image

    from Source.PaintTransformerReference.rendering import load_meta_brushes, param2stroke
    from Source.PaintTransformerReference.synthesize_samples import _save_tensor_image, _stroke_to_paint_transformer_param
    from Source.Renderer.stroke_schema import load_stroke_program_json

    raw_program = _read_json(full_program_path)
    stroke_program = load_stroke_program_json(raw_program)
    if base_count <= 0 or base_count >= len(stroke_program.strokes):
        raise ValueError(f"base_count must split the full program: {base_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_brushes = load_meta_brushes(device)
    canvas = torch.zeros(
        1,
        3,
        stroke_program.canvas.height,
        stroke_program.canvas.width,
        device=device,
    )
    draft_render_dir.mkdir(parents=True, exist_ok=True)
    finished_render_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stroke_index, stroke in enumerate(stroke_program.strokes, start=1):
            param = _stroke_to_paint_transformer_param(torch, stroke).to(device)
            foreground, alpha = param2stroke(param, stroke_program.canvas.height, stroke_program.canvas.width, meta_brushes)
            opacity = torch.tensor(stroke.opacity, device=device).view(1, 1, 1, 1)
            canvas = foreground * alpha * opacity + canvas * (1 - alpha * opacity)
            if stroke_index == base_count:
                _save_tensor_image(canvas[0], draft_render_dir / "final.png", Image)
        _save_tensor_image(canvas[0], finished_render_dir / "final.png", Image)

    _link_or_copy(draft_render_dir / "final.png", draft_image_path)
    _link_or_copy(finished_render_dir / "final.png", finished_image_path)
    _write_final_only_render_manifest(
        render_dir=draft_render_dir,
        program_path=full_program_path,
        canvas=raw_program["canvas"],
        stroke_count=base_count,
        image_completion_ratio=base_count / len(stroke_program.strokes),
    )
    _write_final_only_render_manifest(
        render_dir=finished_render_dir,
        program_path=full_program_path,
        canvas=raw_program["canvas"],
        stroke_count=len(stroke_program.strokes),
        image_completion_ratio=1.0,
    )


def _write_final_only_render_manifest(
    render_dir: Path,
    program_path: Path,
    canvas: dict[str, Any],
    stroke_count: int,
    image_completion_ratio: float,
) -> None:
    _write_json(
        render_dir / "render_manifest.json",
        {
            "version": 1,
            "renderer": "paint_transformer_reference_split_final_only",
            "input": str(program_path),
            "canvas": canvas,
            "stroke_count": stroke_count,
            "frame_count": 1,
            "image_completion_ratio": image_completion_ratio,
            "final_image": str(render_dir / "final.png"),
        },
    )


def _split_sample_dirs(
    sample_dirs: list[Path],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[Path]]:
    total_count = len(sample_dirs)
    test_count = round(total_count * test_fraction)
    val_count = round(total_count * val_fraction)
    train_count = total_count - val_count - test_count
    return {
        "Train": sample_dirs[:train_count],
        "Val": sample_dirs[train_count:train_count + val_count],
        "Test": sample_dirs[train_count + val_count:],
    }


def _program(template_program: dict[str, Any], metadata: dict[str, Any], strokes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": template_program["version"],
        "canvas": template_program["canvas"],
        "metadata": metadata,
        "strokes": strokes,
    }


def _link_or_copy(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        try:
            if source_path.samefile(destination_path):
                return
        except OSError:
            pass
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _assert_same_file_contents(actual_path: Path, expected_path: Path, description: str) -> None:
    if not filecmp.cmp(actual_path, expected_path, shallow=False):
        raise ValueError(f"{description} does not match expected source: {actual_path} != {expected_path}")


def _rmtree_with_retries(path: Path, retries: int = 5, delay_seconds: float = 0.2) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_seconds)


def _validate_args(
    base_count: int | None,
    finishing_count: int | None,
    completion_ratio: float,
    min_completion: float,
    max_completion: float,
    draft_image_completion_ratio: float,
    draft_image_min_completion: float,
    draft_image_max_completion: float,
    val_fraction: float,
    test_fraction: float,
    target_selection_mode: str,
    render_draft_from_base: bool,
    use_output_detail_pair: bool,
) -> None:
    if (base_count is None) != (finishing_count is None):
        raise ValueError("base_count and finishing_count must both be set or both be unset")
    if base_count is not None and base_count <= 0:
        raise ValueError("base_count must be positive")
    if finishing_count is not None and finishing_count <= 0:
        raise ValueError("finishing_count must be positive")
    if not 0.0 < min_completion <= completion_ratio <= max_completion < 1.0:
        raise ValueError("completion values must satisfy 0 < min <= completion <= max < 1")
    if not 0.0 < draft_image_min_completion <= draft_image_completion_ratio <= draft_image_max_completion <= 1.0:
        raise ValueError("draft image completion values must satisfy 0 < min <= completion <= max <= 1")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("split fractions must be non-negative and leave room for Train")
    if target_selection_mode not in (TARGET_SELECTION_ORDERED, TARGET_SELECTION_STRUCTURE_FIRST, TARGET_SELECTION_VISUAL_DELTA):
        raise ValueError(
            "target_selection_mode must be "
            f"{TARGET_SELECTION_ORDERED!r}, {TARGET_SELECTION_STRUCTURE_FIRST!r}, or {TARGET_SELECTION_VISUAL_DELTA!r}"
        )
    if target_selection_mode == TARGET_SELECTION_STRUCTURE_FIRST and (base_count is None or finishing_count is None):
        raise ValueError("structure-first target selection requires --base-count and --finishing-count")
    if target_selection_mode == TARGET_SELECTION_VISUAL_DELTA and not use_output_detail_pair:
        raise ValueError("visual-delta target selection requires --use-output-detail-pair")
    if render_draft_from_base and (base_count is None or finishing_count is None):
        raise ValueError("--render-draft-from-base requires --base-count and --finishing-count")
    if use_output_detail_pair:
        if base_count is not None or finishing_count is not None:
            raise ValueError("--use-output-detail-pair cannot be combined with fixed stroke counts")
        if target_selection_mode not in (TARGET_SELECTION_ORDERED, TARGET_SELECTION_VISUAL_DELTA):
            raise ValueError("--use-output-detail-pair requires ordered or visual-delta target selection")
        if render_draft_from_base:
            raise ValueError("--use-output-detail-pair copies the source PaintTransformer draft automatically")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
