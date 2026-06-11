"""Evaluate a trained BrushWright stroke predictor checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from Source.Model.stroke_dataset import BrushWrightStrokeDataset
from Source.Model.draft_image_encoder import DraftImageEncoderConfig
from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
from Source.Model.stroke_encoder import StrokeEncoderConfig
from Source.Model.stroke_loss import compute_stroke_loss
from Source.Model.stroke_predictor import BrushWrightStrokePredictor
from Source.Model.train_strokes import (
    DEFAULT_CUDA_ATTENTION_BACKEND,
    DEFAULT_DATA_ROOT,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    _build_loader,
    _configure_cuda_attention,
    _move_batch_to_device,
    _resolve_device,
    _resolve_num_workers,
)


DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best.pt"
DEFAULT_SPLIT = "Test"


@dataclass(frozen=True)
class StrokeEvaluationConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint: Path = DEFAULT_CHECKPOINT
    split: str = DEFAULT_SPLIT
    output_path: Path | None = None
    batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    device: str = "auto"
    num_workers: int = DEFAULT_NUM_WORKERS
    numeric_weight: float = 1.0
    brush_weight: float = 0.25
    distribution_weight: float = 0.2
    visual_weight: float = 1.0
    set_matching: bool = True
    cuda_attention_backend: str = DEFAULT_CUDA_ATTENTION_BACKEND
    limit_chunks: int | None = None
    log_every: int = 10


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = evaluate_strokes(
        StrokeEvaluationConfig(
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            split=args.split,
            output_path=args.output_path,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            numeric_weight=args.numeric_weight,
            brush_weight=args.brush_weight,
            distribution_weight=args.distribution_weight,
            visual_weight=args.visual_weight,
            set_matching=not args.no_set_matching,
            cuda_attention_backend=args.cuda_attention_backend,
            limit_chunks=args.limit_chunks,
            log_every=args.log_every,
        )
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


def evaluate_strokes(config: StrokeEvaluationConfig) -> dict[str, Any]:
    _validate_config(config)
    _configure_cuda_attention(config.cuda_attention_backend)
    device = _resolve_device(config.device)
    num_workers = _resolve_num_workers(config.num_workers)
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

    dataset = BrushWrightStrokeDataset(
        config.data_root / config.split,
        chunk_size=decoder_config.chunk_size,
        max_base_strokes=encoder_config.max_strokes,
    )
    if config.limit_chunks is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(config.limit_chunks, len(dataset))))
    loader = _build_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    metrics = _evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        numeric_weight=config.numeric_weight,
        brush_weight=config.brush_weight,
        distribution_weight=config.distribution_weight,
        visual_weight=config.visual_weight,
        set_matching=config.set_matching,
        log_every=config.log_every,
    )
    result = {
        "checkpoint": str(config.checkpoint),
        "split": config.split,
        "chunks": len(dataset),
        "batch_size": config.batch_size,
        "device": str(device),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "loss": metrics["loss"],
        "numeric_loss": metrics["numeric_loss"],
        "brush_loss": metrics["brush_loss"],
        "distribution_loss": metrics["distribution_loss"],
        "visual_loss": metrics["visual_loss"],
        "valid_target_count": metrics["valid_target_count"],
        "batches": metrics["batches"],
    }
    if config.output_path is not None:
        _write_json(config.output_path, result)
    return result


def _evaluate_loader(
    model: BrushWrightStrokePredictor,
    loader,
    device: torch.device,
    numeric_weight: float,
    brush_weight: float,
    distribution_weight: float,
    visual_weight: float,
    set_matching: bool,
    log_every: int,
) -> dict[str, float]:
    total_loss = 0.0
    total_numeric = 0.0
    total_brush = 0.0
    total_distribution = 0.0
    total_visual = 0.0
    total_valid = 0
    batches = 0
    with torch.no_grad():
        loader_length = len(loader)
        for batch_index, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            prediction = model(batch)
            loss = compute_stroke_loss(
                prediction,
                batch,
                numeric_weight=numeric_weight,
                brush_weight=brush_weight,
                distribution_weight=distribution_weight,
                visual_weight=visual_weight,
                set_matching=set_matching,
            )
            weight = max(loss.valid_target_count, 1)
            total_loss += float(loss.total.detach().cpu()) * weight
            total_numeric += float(loss.numeric.detach().cpu()) * weight
            total_brush += float(loss.brush.detach().cpu()) * weight
            total_distribution += float(loss.distribution.detach().cpu()) * weight
            total_visual += float(loss.visual.detach().cpu()) * weight
            total_valid += loss.valid_target_count
            batches += 1
            if _should_log_batch(batch_index, loader_length, log_every):
                denominator = max(total_valid, 1)
                print(
                    f"eval batch {batch_index}/{loader_length} "
                    f"loss={total_loss / denominator:.6f} "
                    f"numeric={total_numeric / denominator:.6f} "
                    f"brush={total_brush / denominator:.6f} "
                    f"distribution={total_distribution / denominator:.6f} "
                    f"visual={total_visual / denominator:.6f} "
                    f"valid={total_valid}",
                    flush=True,
                )
    denominator = max(total_valid, 1)
    return {
        "loss": total_loss / denominator,
        "numeric_loss": total_numeric / denominator,
        "brush_loss": total_brush / denominator,
        "distribution_loss": total_distribution / denominator,
        "visual_loss": total_visual / denominator,
        "valid_target_count": float(total_valid),
        "batches": float(batches),
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise OSError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def _image_encoder_config_from_checkpoint(checkpoint: dict[str, Any]) -> DraftImageEncoderConfig | None:
    config = checkpoint.get("image_encoder_config")
    if config is None:
        return None
    return DraftImageEncoderConfig(**config)


def _validate_config(config: StrokeEvaluationConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.limit_chunks is not None and config.limit_chunks <= 0:
        raise ValueError("limit_chunks must be positive")
    if config.log_every < 0:
        raise ValueError("log_every must be non-negative")
    if config.distribution_weight < 0:
        raise ValueError("distribution_weight must be non-negative")
    if config.visual_weight < 0:
        raise ValueError("visual_weight must be non-negative")
    if config.cuda_attention_backend not in ("math", "default"):
        raise ValueError("cuda_attention_backend must be 'math' or 'default'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a BrushWright stroke predictor checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_DIR / "test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--numeric-weight", type=float, default=1.0)
    parser.add_argument("--brush-weight", type=float, default=0.25)
    parser.add_argument("--distribution-weight", type=float, default=0.2)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-set-matching",
        action="store_true",
        help="Use ordered slot-to-target stroke loss for ablation/debugging.",
    )
    parser.add_argument("--cuda-attention-backend", choices=("math", "default"), default=DEFAULT_CUDA_ATTENTION_BACKEND)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def _should_log_batch(batch_index: int, loader_length: int, log_every: int) -> bool:
    if log_every == 0:
        return batch_index == loader_length
    return batch_index == 1 or batch_index == loader_length or batch_index % log_every == 0


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
