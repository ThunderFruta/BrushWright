"""Export rendered test-set predictions from a trained stroke predictor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch

from Source.Model.draft_image_encoder import DraftImageEncoderConfig
from Source.Model.prediction_diagnostics import compute_prediction_diagnostics
from Source.Model.stroke_dataset import StrokeBatch, load_draft_image_tensor
from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
from Source.Model.stroke_encoder import StrokeEncoderConfig
from Source.Model.stroke_predictor import BrushWrightStrokePredictor
from Source.Model.stroke_tokenizer import NUMERIC_FIELDS, StrokeTokenBatch, StrokeTokenizer
from Source.Model.train_strokes import (
    DEFAULT_CUDA_ATTENTION_BACKEND,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    _configure_cuda_attention,
    _move_batch_to_device,
    _resolve_device,
)
from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.synthesize_samples import render_program_final_with_paint_transformer
from Source.Renderer.stroke_schema import STROKE_PROGRAM_VERSION, load_stroke_program_json


DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best.pt"
DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/TestPredictions")
DEFAULT_LIMIT = 4
DEFAULT_SPLIT = "Test"
DEFAULT_FALLBACK_BRUSH = "paint_transformer_rect"
DEFAULT_MIN_CHANGED_PIXEL_RATIO = 0.0005


@dataclass(frozen=True)
class ExportPredictionsConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint: Path = DEFAULT_CHECKPOINT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    limit: int = DEFAULT_LIMIT
    device: str = "auto"
    cuda_attention_backend: str = DEFAULT_CUDA_ATTENTION_BACKEND
    render: bool = True
    min_changed_pixel_ratio: float = DEFAULT_MIN_CHANGED_PIXEL_RATIO


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exported = export_test_predictions(
        ExportPredictionsConfig(
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            split=args.split,
            limit=args.limit,
            device=args.device,
            cuda_attention_backend=args.cuda_attention_backend,
            render=not args.no_render,
            min_changed_pixel_ratio=args.min_changed_pixel_ratio,
        )
    )
    print(json.dumps({"exported": exported}, indent=2), flush=True)
    return 0


def export_test_predictions(config: ExportPredictionsConfig) -> list[dict[str, Any]]:
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    _configure_cuda_attention(config.cuda_attention_backend)
    device = _resolve_device(config.device)
    checkpoint = _load_checkpoint(config.checkpoint, device)
    encoder_config = StrokeEncoderConfig(**checkpoint["encoder_config"])
    decoder_config = StrokeChunkDecoderConfig(**checkpoint["decoder_config"])
    image_encoder_config = _image_encoder_config_from_checkpoint(checkpoint)
    model = BrushWrightStrokePredictor(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        image_encoder_config=image_encoder_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_root = config.data_root / config.split
    manifest = _read_json(split_root / "dataset_manifest.json")
    samples = manifest.get("samples", [])[: config.limit]
    tokenizer = StrokeTokenizer(
        max_strokes=encoder_config.max_strokes,
        brush_vocab=encoder_config.brush_vocab,
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
                sample_dir=sample_dir,
                sample=sample,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                decoder_config=decoder_config,
                checkpoint=checkpoint,
                checkpoint_path=config.checkpoint,
                device=device,
                render=config.render,
                min_changed_pixel_ratio=config.min_changed_pixel_ratio,
            )
        )
    _write_json(
        prepared_root / "export_manifest.json",
        {
            "version": 1,
            "split": config.split,
            "summary": _export_summary(exported),
            "samples": exported,
        },
    )
    return exported


def _export_sample(
    sample_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    model: BrushWrightStrokePredictor,
    tokenizer: StrokeTokenizer,
    decoder_config: StrokeChunkDecoderConfig,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    render: bool,
    min_changed_pixel_ratio: float,
) -> dict[str, Any]:
    base_program = _read_json(sample_dir / sample["base_strokes"])
    target_program = _read_json(sample_dir / sample["finishing_strokes"])
    draft_image_path = sample_dir / sample["draft_image"]
    target_image_path = _target_image_path_for_prediction(sample_dir, sample, model)
    predicted_finishing = _predict_finishing_strokes(
        model=model,
        tokenizer=tokenizer,
        base_program=base_program,
        draft_image_path=draft_image_path,
        target_image_path=target_image_path,
        finishing_count=int(sample["finishing_count"]),
        decoder_config=decoder_config,
        checkpoint=checkpoint,
        device=device,
    )
    predicted_finishing_program = _program_like(
        template=target_program,
        metadata={
            **dict(target_program.get("metadata", {})),
            "prediction_source": str(checkpoint_path),
            "sample_id": sample["sample_id"],
            "split": "predicted_finishing",
        },
        strokes=predicted_finishing,
    )
    predicted_full_program = _program_like(
        template=base_program,
        metadata={
            **dict(base_program.get("metadata", {})),
            "prediction_source": str(checkpoint_path),
            "sample_id": sample["sample_id"],
            "split": "base_plus_predicted_finishing",
        },
        strokes=[*base_program["strokes"], *predicted_finishing],
    )
    _write_json(output_dir / "predicted_finishing_strokes.json", predicted_finishing_program)
    _write_json(output_dir / "predicted_full_program.json", predicted_full_program)
    _write_json(output_dir / "sample.json", sample | {"prediction_checkpoint": str(checkpoint_path)})
    shutil.copy2(sample_dir / sample["draft_image"], output_dir / "draft.png")
    shutil.copy2(_sample_target_image_path(sample_dir, sample), output_dir / "target.png")

    if render:
        render_program_final_with_paint_transformer(
            output_dir / "predicted_finishing_strokes.json",
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
            predicted_strokes=predicted_finishing,
            target_strokes=target_program["strokes"],
            min_changed_pixel_ratio=min_changed_pixel_ratio,
        )
        _write_json(output_dir / "diagnostics.json", diagnostics)
    else:
        diagnostics = {
            "status": "not_rendered",
            "visual_improved": None,
            "predicted_strokes": {},
            "image_deltas": {},
        }

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
        "predicted_finishing_strokes": str(output_dir / "predicted_finishing_strokes.json"),
        "predicted_full_program": str(output_dir / "predicted_full_program.json"),
    }


def _predict_finishing_strokes(
    model: BrushWrightStrokePredictor,
    tokenizer: StrokeTokenizer,
    base_program: dict[str, Any],
    draft_image_path: Path,
    target_image_path: Path | None,
    finishing_count: int,
    decoder_config: StrokeChunkDecoderConfig,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    base_tokens = tokenizer.encode_program(base_program)
    draft_image = load_draft_image_tensor(draft_image_path).unsqueeze(0)
    target_image = load_draft_image_tensor(target_image_path).unsqueeze(0) if target_image_path is not None else None
    error_map = torch.abs(target_image - draft_image) if target_image is not None else None
    predictions = []
    with torch.no_grad():
        for chunk_start in range(0, finishing_count, decoder_config.chunk_size):
            batch = _prediction_batch(
                base_tokens=base_tokens,
                draft_image=draft_image,
                target_image=target_image,
                error_map=error_map,
                chunk_start=chunk_start,
                chunk_size=decoder_config.chunk_size,
            )
            output = model(_move_batch_to_device(batch, device))
            chunk_strokes = _prediction_to_strokes(
                numeric=output.pred_numeric[0].detach().cpu(),
                brush_logits=output.pred_brush_logits[0].detach().cpu(),
                checkpoint=checkpoint,
            )
            remaining = finishing_count - len(predictions)
            predictions.extend(chunk_strokes[:remaining])
    return predictions[:finishing_count]


def _target_image_path_for_prediction(
    sample_dir: Path,
    sample: dict[str, Any],
    model: BrushWrightStrokePredictor,
) -> Path | None:
    input_channels = None if model.image_encoder is None else model.image_encoder.config.input_channels
    image_name = sample.get("target_image") or sample.get("finished_image")
    if input_channels == 9:
        if not image_name:
            raise ValueError("target-guided export requires sample target_image metadata")
        target_path = sample_dir / image_name
        if not target_path.exists():
            raise ValueError(f"target-guided export requires target image: {target_path}")
        return target_path
    if image_name:
        target_path = sample_dir / image_name
        return target_path if target_path.exists() else None
    return None


def _sample_target_image_path(sample_dir: Path, sample: dict[str, Any]) -> Path:
    image_name = sample.get("target_image") or sample.get("finished_image")
    if not image_name:
        raise ValueError(f"{sample_dir} sample is missing target_image metadata")
    return sample_dir / str(image_name)


def _prediction_batch(
    base_tokens: StrokeTokenBatch,
    draft_image: torch.Tensor,
    target_image: torch.Tensor | None,
    error_map: torch.Tensor | None,
    chunk_start: int,
    chunk_size: int,
) -> StrokeBatch:
    return StrokeBatch(
        base_tokens=base_tokens,
        target_numeric=torch.zeros(1, chunk_size, len(NUMERIC_FIELDS), dtype=torch.float32),
        target_brush_ids=torch.zeros(1, chunk_size, dtype=torch.long),
        target_padding_mask=torch.zeros(1, chunk_size, dtype=torch.bool),
        sample_ids=("prediction",),
        chunk_starts=torch.tensor([chunk_start], dtype=torch.long),
        chunk_ends=torch.tensor([chunk_start + chunk_size], dtype=torch.long),
        stroke_count_adjusted=torch.tensor([False], dtype=torch.bool),
        draft_images=draft_image,
        goal_images=target_image,
        error_maps=error_map,
    )


def _prediction_to_strokes(
    numeric: torch.Tensor,
    brush_logits: torch.Tensor,
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    brush_ids = torch.argmax(brush_logits, dim=-1).tolist()
    return [
        _stroke_from_numeric(numeric[index].tolist(), brush_id=brush_id, checkpoint=checkpoint)
        for index, brush_id in enumerate(brush_ids)
    ]


def _stroke_from_numeric(values: list[float], brush_id: int, checkpoint: dict[str, Any]) -> dict[str, Any]:
    clipped = [max(0.0, min(1.0, float(value))) for value in values]
    brush = _brush_from_id(brush_id, checkpoint)
    return {
        "x": clipped[0],
        "y": clipped[1],
        "angle": clipped[2],
        "length": clipped[3],
        "width": clipped[4],
        "opacity": clipped[5],
        "color": [clipped[6], clipped[7], clipped[8]],
        "brush": brush,
    }


def _brush_from_id(brush_id: int, checkpoint: dict[str, Any]) -> str:
    id_to_brush = checkpoint.get("tokenizer", {}).get("id_to_brush", {})
    brush = id_to_brush.get(brush_id, id_to_brush.get(str(brush_id)))
    if not brush or brush in ("<PAD>", "<UNK>"):
        return DEFAULT_FALLBACK_BRUSH
    return str(brush)


def _program_like(template: dict[str, Any], metadata: dict[str, Any], strokes: list[dict[str, Any]]) -> dict[str, Any]:
    program = {
        "version": template.get("version", STROKE_PROGRAM_VERSION),
        "canvas": template["canvas"],
        "metadata": metadata,
        "strokes": strokes,
    }
    load_stroke_program_json(program)
    return program


def _write_comparison_strip(draft_path: Path, target_path: Path, predicted_path: Path, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in (draft_path, target_path, predicted_path)]
    width, height = images[0].size
    label_height = 28
    canvas = Image.new("RGB", (width * 3, height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(("draft", "target", "predicted"), images)):
        canvas.paste(image, (index * width, label_height))
        draw.text((index * width + 8, 8), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise OSError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def _image_encoder_config_from_checkpoint(checkpoint: dict[str, Any]) -> DraftImageEncoderConfig | None:
    config = checkpoint.get("image_encoder_config")
    if config is None:
        return None
    return DraftImageEncoderConfig(**config)


def _export_summary(exported: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [entry for entry in exported if entry.get("visual_improved") is not None]
    improved = [entry for entry in rendered if entry.get("visual_improved")]
    low_change = [entry for entry in rendered if entry.get("status") == "failed_low_pixel_change"]
    return {
        "sample_count": len(exported),
        "rendered_count": len(rendered),
        "improved_count": len(improved),
        "visual_improvement_rate": len(improved) / len(rendered) if rendered else 0.0,
        "low_change_count": len(low_change),
        "low_change_rate": len(low_change) / len(rendered) if rendered else 0.0,
        "checkpoint_status": "visual_pass" if improved else "visual_failed",
        "status_histogram": _status_histogram(exported),
    }


def _status_histogram(exported: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entry in exported:
        status = str(entry.get("status", "unknown"))
        histogram[status] = histogram.get(status, 0) + 1
    return dict(sorted(histogram.items()))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export rendered BrushWright test predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-attention-backend", choices=("math", "default"), default=DEFAULT_CUDA_ATTENTION_BACKEND)
    parser.add_argument("--min-changed-pixel-ratio", type=float, default=DEFAULT_MIN_CHANGED_PIXEL_RATIO)
    parser.add_argument("--no-render", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
