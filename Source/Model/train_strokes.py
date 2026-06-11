"""Train the stroke-only BrushWright predictor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from Source.Model.draft_image_encoder import DraftImageEncoderConfig
from Source.Model.stroke_dataset import BrushWrightStrokeDataset, StrokeBatch, collate_stroke_chunks
from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
from Source.Model.stroke_encoder import StrokeEncoderConfig
from Source.Model.stroke_loss import compute_stroke_loss
from Source.Model.stroke_predictor import BrushWrightStrokePredictor


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_OUTPUT_DIR = Path("Models/Checkpoints/StrokePredictorV1TargetGuided")
DEFAULT_BATCH_SIZE = 16
DEFAULT_MICRO_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_CUDA_ATTENTION_BACKEND = "math"
DEFAULT_CHECKPOINT_EVERY_STEPS = 25


@dataclass(frozen=True)
class StrokeTrainingConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 5
    batch_size: int = DEFAULT_BATCH_SIZE
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    device: str = "auto"
    num_workers: int = DEFAULT_NUM_WORKERS
    model_dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    chunk_size: int = 64
    max_base_strokes: int = 192
    max_chunks: int = 24
    image_grid_size: int = 8
    image_input_channels: int = 9
    decoder_query_mode: str = "learned"
    require_v1_contract: bool = True
    overfit_samples: int = 0
    train_repeat_factor: int = 1
    numeric_weight: float = 1.0
    brush_weight: float = 0.25
    distribution_weight: float = 0.2
    visual_weight: float = 1.0
    set_matching: bool = True
    visual_validation_samples: int = 4
    visual_validation_device: str = "cpu"
    visual_validation_interval: int = 1
    min_visual_changed_pixel_ratio: float = 0.0005
    seed: int = 20260602
    log_every: int = 10
    cuda_attention_backend: str = DEFAULT_CUDA_ATTENTION_BACKEND
    checkpoint_every_steps: int = DEFAULT_CHECKPOINT_EVERY_STEPS
    resume_checkpoint: Path | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = StrokeTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        num_workers=args.num_workers,
        model_dim=args.model_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
        max_base_strokes=args.max_base_strokes,
        max_chunks=args.max_chunks,
        image_grid_size=args.image_grid_size,
        image_input_channels=args.image_input_channels,
        decoder_query_mode=args.decoder_query_mode,
        require_v1_contract=not args.no_require_v1_contract,
        overfit_samples=args.overfit_samples,
        train_repeat_factor=args.train_repeat_factor,
        numeric_weight=args.numeric_weight,
        brush_weight=args.brush_weight,
        distribution_weight=args.distribution_weight,
        visual_weight=args.visual_weight,
        set_matching=not args.no_set_matching,
        visual_validation_samples=args.visual_validation_samples,
        visual_validation_device=args.visual_validation_device,
        visual_validation_interval=args.visual_validation_interval,
        min_visual_changed_pixel_ratio=args.min_visual_changed_pixel_ratio,
        seed=args.seed,
        log_every=args.log_every,
        cuda_attention_backend=args.cuda_attention_backend,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_checkpoint=args.resume_checkpoint,
    )
    try:
        train_strokes(config)
    except Exception as exc:
        if _is_cuda_oom(exc):
            print(
                "train failed: CUDA ran out of memory. "
                "Keep `--batch-size` as the effective batch, but lower `--micro-batch-size` so each GPU step is smaller. "
                "Try `--micro-batch-size 8`, or `--micro-batch-size 4` if 8 still fails.",
                file=sys.stderr,
            )
            print(f"cuda error detail: {exc}", file=sys.stderr)
            return 1
        if _is_cuda_failure(exc):
            print(
                "train failed: CUDA reported a kernel/runtime failure. "
                "This is usually a driver, PyTorch CUDA build, or GPU architecture mismatch, not the metric line itself. "
                "Try `--device cpu` to continue on CPU, or rerun with `CUDA_LAUNCH_BLOCKING=1` for a precise CUDA stack.",
                file=sys.stderr,
            )
            print(f"cuda error detail: {exc}", file=sys.stderr)
            return 1
        raise
    return 0


def train_strokes(config: StrokeTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    _configure_cuda_attention(config.cuda_attention_backend)
    device = _resolve_device(config.device)
    micro_batch_size = min(config.micro_batch_size, config.batch_size)
    accumulation_steps = max(1, (config.batch_size + micro_batch_size - 1) // micro_batch_size)
    num_workers = _resolve_num_workers(config.num_workers)
    print("BrushWright stroke training", flush=True)
    print(f"  data root: {config.data_root}", flush=True)
    print(f"  output dir: {config.output_dir}", flush=True)
    print(f"  requested device: {config.device}", flush=True)
    print(f"  resolved device: {device}", flush=True)
    print(
        f"  model: dim={config.model_dim} encoder_layers={config.encoder_layers} "
        f"decoder_layers={config.decoder_layers} heads={config.num_heads}",
        flush=True,
    )
    print(
        f"  training: epochs={config.epochs} effective_batch_size={config.batch_size} "
        f"micro_batch_size={micro_batch_size} accumulation_steps={accumulation_steps} "
        f"lr={config.learning_rate} weight_decay={config.weight_decay}",
        flush=True,
    )
    print(
        f"  checkpointing: every {config.checkpoint_every_steps} optimizer step(s); "
        f"resume={config.resume_checkpoint or 'none'}",
        flush=True,
    )
    print(f"  cuda attention backend: {config.cuda_attention_backend}", flush=True)
    print(
        f"  data loading: requested_workers={config.num_workers} num_workers={num_workers} "
        f"persistent_workers={num_workers > 0} pin_memory={device.type == 'cuda'} cache_samples=True",
        flush=True,
    )
    print("Loading datasets...", flush=True)
    full_train_dataset = BrushWrightStrokeDataset(
        config.data_root / "Train",
        chunk_size=config.chunk_size,
        max_base_strokes=config.max_base_strokes,
        require_v1_contract=config.require_v1_contract,
    )
    full_val_dataset = BrushWrightStrokeDataset(
        config.data_root / "Val",
        chunk_size=config.chunk_size,
        max_base_strokes=config.max_base_strokes,
        require_v1_contract=config.require_v1_contract,
    )
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset
    if config.overfit_samples:
        count = min(config.overfit_samples, len(full_train_dataset))
        indices = list(range(count))
        train_dataset = Subset(full_train_dataset, indices)
        val_dataset = Subset(full_train_dataset, indices)
        print(f"  overfit mode: first {count} Train chunk(s)", flush=True)
    if config.train_repeat_factor > 1:
        train_dataset = ConcatDataset([train_dataset] * config.train_repeat_factor)
        print(f"  train repeat factor: {config.train_repeat_factor}", flush=True)
    print(
        f"  Train chunks: {len(train_dataset)} from {config.data_root / 'Train'}",
        flush=True,
    )
    print(
        f"  Val chunks: {len(val_dataset)} from "
        f"{(config.data_root / 'Train') if config.overfit_samples else (config.data_root / 'Val')}",
        flush=True,
    )

    train_loader = _build_loader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=not bool(config.overfit_samples),
        num_workers=num_workers,
        device=device,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    encoder_config = StrokeEncoderConfig(
        model_dim=config.model_dim,
        num_layers=config.encoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_strokes=config.max_base_strokes,
    )
    decoder_config = StrokeChunkDecoderConfig(
        model_dim=config.model_dim,
        num_layers=config.decoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        chunk_size=config.chunk_size,
        max_chunks=config.max_chunks,
        query_mode=config.decoder_query_mode,
        spatial_grid_size=config.image_grid_size,
    )
    image_encoder_config = DraftImageEncoderConfig(
        model_dim=config.model_dim,
        grid_size=config.image_grid_size,
        dropout=config.dropout,
        input_channels=config.image_input_channels,
    )
    print("Building model and optimizer...", flush=True)
    model = BrushWrightStrokePredictor(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        image_encoder_config=image_encoder_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing checkpoints to {config.output_dir}", flush=True)
    metrics: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_visual_improvement_rate = -1.0
    start_epoch = 1
    global_step = 0
    if config.resume_checkpoint is not None:
        start_epoch, global_step, metrics, best_val_loss, best_visual_improvement_rate = _load_training_checkpoint(
            path=config.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Resumed checkpoint {config.resume_checkpoint} at epoch {start_epoch}, global_step {global_step}",
            flush=True,
        )

    def checkpoint_step(epoch: int) -> int:
        nonlocal global_step
        global_step += 1
        if config.checkpoint_every_steps > 0 and global_step % config.checkpoint_every_steps == 0:
            _save_checkpoint(
                path=config.output_dir / "step_latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"epoch": epoch, "global_step": global_step, "type": "step"},
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                image_encoder_config=image_encoder_config,
                train_dataset=train_dataset,
                checkpoint_type="step",
                global_step=global_step,
                metrics_log=metrics,
            )
            print(f"  wrote {config.output_dir / 'step_latest.pt'} at optimizer step {global_step}", flush=True)
        return global_step

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"epoch {epoch}/{config.epochs} train start", flush=True)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            numeric_weight=config.numeric_weight,
            brush_weight=config.brush_weight,
            distribution_weight=config.distribution_weight,
            visual_weight=config.visual_weight,
            set_matching=config.set_matching,
            epoch=epoch,
            phase="train",
            log_every=config.log_every,
            accumulation_steps=accumulation_steps,
            on_optimizer_step=checkpoint_step,
        )
        print(f"epoch {epoch}/{config.epochs} val start", flush=True)
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            numeric_weight=config.numeric_weight,
            brush_weight=config.brush_weight,
            distribution_weight=config.distribution_weight,
            visual_weight=config.visual_weight,
            set_matching=config.set_matching,
            epoch=epoch,
            phase="val",
            log_every=config.log_every,
            accumulation_steps=1,
            on_optimizer_step=None,
        )
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics.append(epoch_metrics)
        print(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}",
            flush=True,
        )
        _save_checkpoint(
            path=config.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_metrics,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            image_encoder_config=image_encoder_config,
            train_dataset=train_dataset,
            checkpoint_type="epoch",
            global_step=global_step,
            metrics_log=metrics,
        )
        print(f"  wrote {config.output_dir / 'latest.pt'}", flush=True)
        visual_metrics = _run_visual_validation(
            config=config,
            checkpoint_path=config.output_dir / "latest.pt",
            epoch=epoch,
        )
        if visual_metrics is not None:
            epoch_metrics["visual"] = visual_metrics
            _write_json(config.output_dir / "visual_metrics.json", _visual_metrics_log(metrics))
            _save_checkpoint(
                path=config.output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                image_encoder_config=image_encoder_config,
                train_dataset=train_dataset,
                checkpoint_type="epoch",
                global_step=global_step,
                metrics_log=metrics,
            )
            print(
                f"  visual validation: improvement_rate={visual_metrics['visual_improvement_rate']:.3f} "
                f"low_change_rate={visual_metrics['low_change_rate']:.3f}",
                flush=True,
            )
        is_better, best_val_loss, best_visual_improvement_rate = _is_better_checkpoint(
            val_loss=val_metrics["loss"],
            visual_metrics=visual_metrics,
            best_val_loss=best_val_loss,
            best_visual_improvement_rate=best_visual_improvement_rate,
        )
        if is_better:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(
                path=config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                image_encoder_config=image_encoder_config,
                train_dataset=train_dataset,
                checkpoint_type="best",
                global_step=global_step,
                metrics_log=metrics,
            )
            print(f"  wrote {config.output_dir / 'best.pt'}", flush=True)
        _write_json(config.output_dir / "metrics.json", metrics)
        print(f"  wrote {config.output_dir / 'metrics.json'}", flush=True)

    return {
        "epochs": metrics,
        "best_val_loss": best_val_loss,
        "output_dir": str(config.output_dir),
        "train_chunk_count": len(train_dataset),
        "val_chunk_count": len(val_dataset),
    }


def _run_epoch(
    model: BrushWrightStrokePredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    numeric_weight: float,
    brush_weight: float,
    distribution_weight: float,
    visual_weight: float,
    set_matching: bool,
    epoch: int,
    phase: str,
    log_every: int,
    accumulation_steps: int,
    on_optimizer_step: Callable[[int], int] | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_numeric = 0.0
    total_brush = 0.0
    total_distribution = 0.0
    total_visual = 0.0
    total_valid = 0
    batches = 0
    optimizer_steps = 0
    loader_length = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
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
            if training:
                group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
                group_end = min(group_start + accumulation_steps - 1, loader_length)
                group_size = group_end - group_start + 1
                (loss.total / group_size).backward()
        total_loss += float(loss.total.detach().cpu()) * max(loss.valid_target_count, 1)
        total_numeric += float(loss.numeric.detach().cpu()) * max(loss.valid_target_count, 1)
        total_brush += float(loss.brush.detach().cpu()) * max(loss.valid_target_count, 1)
        total_distribution += float(loss.distribution.detach().cpu()) * max(loss.valid_target_count, 1)
        total_visual += float(loss.visual.detach().cpu()) * max(loss.valid_target_count, 1)
        total_valid += loss.valid_target_count
        batches += 1
        if training and (batch_index % accumulation_steps == 0 or batch_index == loader_length):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if on_optimizer_step is not None:
                on_optimizer_step(epoch)
        if _should_log_batch(batch_index, loader_length, log_every):
            denominator = max(total_valid, 1)
            print(
                f"epoch {epoch} {phase} batch {batch_index}/{loader_length} "
                f"loss={total_loss / denominator:.6f} "
                f"numeric={total_numeric / denominator:.6f} "
                f"brush={total_brush / denominator:.6f} "
                f"distribution={total_distribution / denominator:.6f} "
                f"visual={total_visual / denominator:.6f} "
                f"valid={total_valid} optimizer_steps={optimizer_steps}",
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
        "optimizer_steps": float(optimizer_steps),
    }


def _build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_stroke_chunks,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _move_batch_to_device(batch: StrokeBatch, device: torch.device) -> StrokeBatch:
    non_blocking = device.type == "cuda"
    return StrokeBatch(
        base_tokens=type(batch.base_tokens)(
            numeric=batch.base_tokens.numeric.to(device, non_blocking=non_blocking),
            brush_ids=batch.base_tokens.brush_ids.to(device, non_blocking=non_blocking),
            padding_mask=batch.base_tokens.padding_mask.to(device, non_blocking=non_blocking),
            lengths=batch.base_tokens.lengths.to(device, non_blocking=non_blocking),
        ),
        target_numeric=batch.target_numeric.to(device, non_blocking=non_blocking),
        target_brush_ids=batch.target_brush_ids.to(device, non_blocking=non_blocking),
        target_padding_mask=batch.target_padding_mask.to(device, non_blocking=non_blocking),
        sample_ids=batch.sample_ids,
        chunk_starts=batch.chunk_starts.to(device, non_blocking=non_blocking),
        chunk_ends=batch.chunk_ends.to(device, non_blocking=non_blocking),
        stroke_count_adjusted=batch.stroke_count_adjusted.to(device, non_blocking=non_blocking),
        draft_images=batch.draft_images.to(device, non_blocking=non_blocking) if batch.draft_images is not None else None,
        goal_images=batch.goal_images.to(device, non_blocking=non_blocking) if batch.goal_images is not None else None,
        error_maps=batch.error_maps.to(device, non_blocking=non_blocking) if batch.error_maps is not None else None,
    )


def _save_checkpoint(
    path: Path,
    model: BrushWrightStrokePredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    encoder_config: StrokeEncoderConfig,
    decoder_config: StrokeChunkDecoderConfig,
    image_encoder_config: DraftImageEncoderConfig | None,
    train_dataset,
    checkpoint_type: str,
    global_step: int,
    metrics_log: list[dict[str, Any]],
) -> None:
    base_dataset = _base_dataset(train_dataset)
    tokenizer = base_dataset.tokenizer
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "encoder_config": asdict(encoder_config),
            "decoder_config": asdict(decoder_config),
            "image_encoder_config": asdict(image_encoder_config) if image_encoder_config is not None else None,
            "checkpoint_type": checkpoint_type,
            "global_step": global_step,
            "metrics_log": metrics_log,
            "tokenizer": {
                "brush_to_id": tokenizer.brush_to_id,
                "id_to_brush": tokenizer.id_to_brush,
                "max_strokes": tokenizer.max_strokes,
                "numeric_dim": tokenizer.numeric_dim,
            },
        },
        path,
    )


def _base_dataset(dataset):
    if isinstance(dataset, Subset):
        return _base_dataset(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return _base_dataset(dataset.datasets[0])
    return dataset


def _load_training_checkpoint(
    path: Path,
    model: BrushWrightStrokePredictor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, list[dict[str, Any]], float, float]:
    if not path.exists():
        raise OSError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint.get("epoch", 0))
    checkpoint_type = str(checkpoint.get("checkpoint_type", "epoch"))
    start_epoch = epoch + 1 if checkpoint_type in ("epoch", "best") else max(1, epoch)
    global_step = int(checkpoint.get("global_step", 0))
    metrics_log = list(checkpoint.get("metrics_log", []))
    best_val_loss = _best_val_loss(metrics_log)
    best_visual_improvement_rate = _best_visual_improvement_rate(metrics_log)
    return start_epoch, global_step, metrics_log, best_val_loss, best_visual_improvement_rate


def _best_val_loss(metrics_log: list[dict[str, Any]]) -> float:
    values = []
    for entry in metrics_log:
        val_metrics = entry.get("val") if isinstance(entry, dict) else None
        if isinstance(val_metrics, dict) and "loss" in val_metrics:
            values.append(float(val_metrics["loss"]))
    return min(values) if values else float("inf")


def _best_visual_improvement_rate(metrics_log: list[dict[str, Any]]) -> float:
    values = []
    for entry in metrics_log:
        visual_metrics = entry.get("visual") if isinstance(entry, dict) else None
        if isinstance(visual_metrics, dict) and "visual_improvement_rate" in visual_metrics:
            values.append(float(visual_metrics["visual_improvement_rate"]))
    return max(values) if values else -1.0


def _run_visual_validation(
    config: StrokeTrainingConfig,
    checkpoint_path: Path,
    epoch: int,
) -> dict[str, Any] | None:
    if config.visual_validation_samples <= 0:
        return None
    if epoch % config.visual_validation_interval != 0 and epoch != config.epochs:
        return None

    from Source.Model.export_test_predictions import ExportPredictionsConfig, export_test_predictions

    split = "Train" if config.overfit_samples else "Val"
    output_root = config.output_dir / "VisualValidation" / f"epoch_{epoch:04d}"
    exported = export_test_predictions(
        ExportPredictionsConfig(
            data_root=config.data_root,
            checkpoint=checkpoint_path,
            output_root=output_root,
            split=split,
            limit=config.visual_validation_samples,
            device=config.visual_validation_device,
            cuda_attention_backend=config.cuda_attention_backend,
            render=True,
            min_changed_pixel_ratio=config.min_visual_changed_pixel_ratio,
        )
    )
    rendered = [entry for entry in exported if entry.get("visual_improved") is not None]
    improved = [entry for entry in rendered if entry.get("visual_improved")]
    low_change = [entry for entry in rendered if entry.get("status") == "failed_low_pixel_change"]
    summary = {
        "epoch": epoch,
        "split": split,
        "output_root": str(output_root),
        "sample_count": len(exported),
        "rendered_count": len(rendered),
        "improved_count": len(improved),
        "visual_improvement_rate": len(improved) / len(rendered) if rendered else 0.0,
        "low_change_count": len(low_change),
        "low_change_rate": len(low_change) / len(rendered) if rendered else 0.0,
        "checkpoint_status": "visual_pass" if improved else "visual_failed",
        "status_histogram": _status_histogram(exported),
    }
    _write_json(output_root / "visual_summary.json", summary)
    return summary


def _status_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        histogram[status] = histogram.get(status, 0) + 1
    return dict(sorted(histogram.items()))


def _visual_metrics_log(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"epoch": entry["epoch"], "visual": entry["visual"]}
        for entry in metrics
        if isinstance(entry, dict) and isinstance(entry.get("visual"), dict)
    ]


def _is_better_checkpoint(
    val_loss: float,
    visual_metrics: dict[str, Any] | None,
    best_val_loss: float,
    best_visual_improvement_rate: float,
) -> tuple[bool, float, float]:
    if visual_metrics is None:
        return val_loss < best_val_loss, min(best_val_loss, val_loss), best_visual_improvement_rate

    visual_rate = float(visual_metrics.get("visual_improvement_rate", 0.0))
    if visual_rate > best_visual_improvement_rate:
        return True, val_loss, visual_rate
    if visual_rate == best_visual_improvement_rate and val_loss < best_val_loss:
        return True, val_loss, visual_rate
    return False, best_val_loss, best_visual_improvement_rate


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        cuda_device = torch.device("cuda")
        if _cuda_training_preflight(cuda_device):
            return cuda_device
        print(
            "CUDA is visible but failed a tiny transformer training preflight; falling back to CPU because --device auto was used.",
            flush=True,
        )
        return torch.device("cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    device = torch.device(device_name)
    if device.type == "cuda" and not _cuda_training_preflight(device):
        raise RuntimeError(
            "CUDA was requested, but a tiny transformer training preflight failed. "
            "Run with --device cpu, or debug CUDA with CUDA_LAUNCH_BLOCKING=1."
        )
    return device


def _configure_cuda_attention(backend: str) -> None:
    if backend == "default":
        return
    if backend != "math":
        raise ValueError(f"unsupported cuda attention backend: {backend}")
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    _call_if_present(cuda_backend, "enable_flash_sdp", False)
    _call_if_present(cuda_backend, "enable_mem_efficient_sdp", False)
    _call_if_present(cuda_backend, "enable_cudnn_sdp", False)
    _call_if_present(cuda_backend, "enable_math_sdp", True)


def _call_if_present(target: object, name: str, value: bool) -> None:
    function = getattr(target, name, None)
    if function is not None:
        function(value)


def _cuda_training_preflight(device: torch.device) -> bool:
    try:
        module = torch.nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        ).to(device)
        inputs = torch.randn(2, 4, 16, device=device, requires_grad=True)
        outputs = module(inputs)
        loss = outputs.square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        print(f"CUDA preflight failed: {exc}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False


def _is_cuda_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "cuda" in message
        or "cublas" in message
        or "cudnn" in message
        or exc.__class__.__name__.lower() in {"acceleratorerror", "cudaerror"}
    )


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _resolve_num_workers(requested_workers: int) -> int:
    if requested_workers <= 0:
        return 0
    if _multiprocessing_listener_available():
        return requested_workers
    print(
        "multiprocessing DataLoader workers are not available in this environment; "
        "falling back to --num-workers 0",
        flush=True,
    )
    return 0


def _multiprocessing_listener_available() -> bool:
    try:
        from multiprocessing.connection import Listener

        listener = Listener(authkey=b"brushwright")
        listener.close()
        return True
    except OSError:
        return False


def _validate_config(config: StrokeTrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.overfit_samples < 0:
        raise ValueError("overfit_samples must be non-negative")
    if config.train_repeat_factor <= 0:
        raise ValueError("train_repeat_factor must be positive")
    if config.log_every < 0:
        raise ValueError("log_every must be non-negative")
    if config.cuda_attention_backend not in ("math", "default"):
        raise ValueError("cuda_attention_backend must be 'math' or 'default'")
    if config.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    if config.image_grid_size <= 0:
        raise ValueError("image_grid_size must be positive")
    if config.image_input_channels not in (3, 9):
        raise ValueError("image_input_channels must be 3 or 9")
    if config.decoder_query_mode not in ("learned", "spatial"):
        raise ValueError("decoder_query_mode must be 'learned' or 'spatial'")
    if config.distribution_weight < 0:
        raise ValueError("distribution_weight must be non-negative")
    if config.visual_weight < 0:
        raise ValueError("visual_weight must be non-negative")
    if config.visual_validation_samples < 0:
        raise ValueError("visual_validation_samples must be non-negative")
    if config.visual_validation_interval <= 0:
        raise ValueError("visual_validation_interval must be positive")
    if config.min_visual_changed_pixel_ratio < 0:
        raise ValueError("min_visual_changed_pixel_ratio must be non-negative")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BrushWright stroke-only predictor.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=DEFAULT_MICRO_BATCH_SIZE,
        help="Actual DataLoader/GPU batch size. Gradients accumulate until --batch-size is reached.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-base-strokes", type=int, default=192)
    parser.add_argument("--max-chunks", type=int, default=24)
    parser.add_argument("--image-grid-size", type=int, default=8)
    parser.add_argument("--image-input-channels", type=int, default=9)
    parser.add_argument("--decoder-query-mode", choices=("learned", "spatial"), default="learned")
    parser.add_argument(
        "--no-require-v1-contract",
        action="store_true",
        help="Allow non-V1 data contracts. Intended only for legacy tests and experiments.",
    )
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--train-repeat-factor", type=int, default=1)
    parser.add_argument("--numeric-weight", type=float, default=1.0)
    parser.add_argument("--brush-weight", type=float, default=0.25)
    parser.add_argument("--distribution-weight", type=float, default=0.2)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-set-matching",
        action="store_true",
        help="Use ordered slot-to-target stroke loss. Intended only for ablation/debugging.",
    )
    parser.add_argument("--visual-validation-samples", type=int, default=4)
    parser.add_argument("--visual-validation-device", default="cpu")
    parser.add_argument("--visual-validation-interval", type=int, default=1)
    parser.add_argument("--min-visual-changed-pixel-ratio", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N train/val batches. Use 0 to only log epoch boundaries.",
    )
    parser.add_argument(
        "--cuda-attention-backend",
        choices=("math", "default"),
        default=DEFAULT_CUDA_ATTENTION_BACKEND,
        help="CUDA attention backend. 'math' is slower but avoids flash/memory-efficient/cuDNN SDP kernels.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
        help="Write step_latest.pt every N optimizer steps. Use 0 to disable step checkpoints.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a checkpoint path.",
    )
    return parser


def _should_log_batch(batch_index: int, loader_length: int, log_every: int) -> bool:
    if log_every == 0:
        return batch_index == loader_length
    return batch_index == 1 or batch_index == loader_length or batch_index % log_every == 0


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
