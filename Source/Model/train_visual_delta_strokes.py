"""Train the BrushWright visual-delta stroke compiler."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from Source.Model.train_strokes import (
    DEFAULT_CUDA_ATTENTION_BACKEND,
    _configure_cuda_attention,
    _resolve_device,
    _resolve_num_workers,
    _should_log_batch,
    _status_histogram,
)
from Source.Model.visual_delta_dataset import (
    TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET,
    VisualDeltaStrokeDataset,
    collate_visual_delta_patches,
    visual_delta_batch_to_device,
)
from Source.Model.visual_delta_loss import (
    SUPPORTED_TRAINING_RENDERERS,
    TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
    compute_visual_delta_loss,
)
from Source.Model.visual_delta_predictor import (
    DEFAULT_FF_DIM,
    DEFAULT_GRID_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_STROKES,
    DEFAULT_MODEL_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    VisualDeltaStrokeCompiler,
    VisualDeltaStrokeCompilerConfig,
)


DEFAULT_DATA_ROOT = Path("Data")
DEFAULT_OUTPUT_DIR = Path("Models/Checkpoints/VisualDeltaStrokeCompilerV8UsableV1Large")
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 8
DEFAULT_EPOCHS = 80
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_PATCH_SIZE = 128
DEFAULT_PATCH_STRIDE = 64
DEFAULT_TARGET_VRAM_GB = 48
DEFAULT_PRESENT_THRESHOLD = 0.05
DEFAULT_PRESENT_POSITIVE_WEIGHT = 8.0


@dataclass(frozen=True)
class VisualDeltaTrainingConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = 0.01
    device: str = "auto"
    num_workers: int = DEFAULT_NUM_WORKERS
    model_dim: int = DEFAULT_MODEL_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    decoder_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    ff_dim: int = DEFAULT_FF_DIM
    dropout: float = 0.1
    patch_size: int = DEFAULT_PATCH_SIZE
    patch_stride: int = DEFAULT_PATCH_STRIDE
    grid_size: int = DEFAULT_GRID_SIZE
    max_strokes_per_patch: int = DEFAULT_MAX_STROKES
    mask_threshold: float = 0.04
    min_changed_pixels: int = 16
    negative_patch_ratio: float = 0.25
    edge_focused_sampling: bool = True
    include_zero_target_changed_patches: bool = True
    overfit_patches: int = 0
    overfit_samples: int = 0
    train_repeat_factor: int = 1
    numeric_weight: float = 1.0
    brush_weight: float = 0.25
    present_weight: float = 1.0
    present_positive_weight: float = DEFAULT_PRESENT_POSITIVE_WEIGHT
    count_weight: float = 0.5
    image_weight: float = 4.0
    preservation_weight: float = 1.0
    gradient_weight: float = 2.0
    edge_weight: float = 1.0
    low_frequency_weight: float = 1.0
    coarse_structure_weight: float = 2.0
    recall_weight: float = 0.5
    anti_dot_weight: float = 0.75
    color_clamp_weight: float = 1.0
    size_distribution_weight: float = 0.5
    slot_aware_targets: bool = True
    training_renderer: str = TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT
    require_structure_targets: bool = False
    require_target_contract: str | None = TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET
    visual_validation_samples: int = 8
    visual_validation_device: str = "cpu"
    visual_validation_interval: int = 1
    min_visual_changed_pixel_ratio: float = 0.005
    min_visual_gradient_improvement: float = 0.0
    min_visual_edge_overlap: float = 0.02
    present_threshold: float = DEFAULT_PRESENT_THRESHOLD
    min_export_candidates_per_sample: int = 0
    max_export_strokes_per_sample: int = DEFAULT_MAX_STROKES
    max_export_strokes_per_patch: int = DEFAULT_MAX_STROKES
    min_export_render_area: float = 8.0
    export_ranking_mode: str = "visual-delta"
    target_vram_gb: int = DEFAULT_TARGET_VRAM_GB
    seed: int = 20260603
    log_every: int = 10
    cuda_attention_backend: str = DEFAULT_CUDA_ATTENTION_BACKEND
    early_stop_zero_selected_epochs: int = 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = VisualDeltaTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        num_workers=args.num_workers,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        grid_size=args.grid_size,
        max_strokes_per_patch=args.max_strokes_per_patch,
        mask_threshold=args.mask_threshold,
        min_changed_pixels=args.min_changed_pixels,
        negative_patch_ratio=args.negative_patch_ratio,
        edge_focused_sampling=not args.no_edge_focused_sampling,
        include_zero_target_changed_patches=args.include_zero_target_changed_patches,
        overfit_patches=args.overfit_patches,
        overfit_samples=args.overfit_samples,
        train_repeat_factor=args.train_repeat_factor,
        numeric_weight=args.numeric_weight,
        brush_weight=args.brush_weight,
        present_weight=args.present_weight,
        present_positive_weight=args.present_positive_weight,
        count_weight=args.count_weight,
        image_weight=args.image_weight,
        preservation_weight=args.preservation_weight,
        gradient_weight=args.gradient_weight,
        edge_weight=args.edge_weight,
        low_frequency_weight=args.low_frequency_weight,
        coarse_structure_weight=args.coarse_structure_weight,
        recall_weight=args.recall_weight,
        anti_dot_weight=args.anti_dot_weight,
        color_clamp_weight=args.color_clamp_weight,
        size_distribution_weight=args.size_distribution_weight,
        slot_aware_targets=args.slot_aware_targets,
        training_renderer=args.training_renderer,
        require_structure_targets=args.require_structure_targets,
        require_target_contract=args.require_target_contract,
        visual_validation_samples=args.visual_validation_samples,
        visual_validation_device=args.visual_validation_device,
        visual_validation_interval=args.visual_validation_interval,
        min_visual_changed_pixel_ratio=args.min_visual_changed_pixel_ratio,
        min_visual_gradient_improvement=args.min_visual_gradient_improvement,
        min_visual_edge_overlap=args.min_visual_edge_overlap,
        present_threshold=args.present_threshold,
        min_export_candidates_per_sample=args.min_export_candidates_per_sample,
        max_export_strokes_per_sample=args.max_export_strokes_per_sample,
        max_export_strokes_per_patch=args.max_export_strokes_per_patch,
        min_export_render_area=args.min_export_render_area,
        export_ranking_mode=args.export_ranking_mode,
        target_vram_gb=args.target_vram_gb,
        seed=args.seed,
        log_every=args.log_every,
        cuda_attention_backend=args.cuda_attention_backend,
        early_stop_zero_selected_epochs=args.early_stop_zero_selected_epochs,
    )
    train_visual_delta_strokes(config)
    return 0


def train_visual_delta_strokes(config: VisualDeltaTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    _configure_cuda_attention(config.cuda_attention_backend)
    device = _resolve_device(config.device)
    num_workers = _resolve_num_workers(config.num_workers)
    print("BrushWright visual-delta stroke compiler training", flush=True)
    print(f"  data root: {config.data_root}", flush=True)
    print(f"  output dir: {config.output_dir}", flush=True)
    print(f"  resolved device: {device}", flush=True)
    print(
        f"  model: dim={config.model_dim} decoder_layers={config.decoder_layers} heads={config.num_heads}",
        flush=True,
    )
    print(f"  target VRAM class: ~{config.target_vram_gb}GB", flush=True)
    print(
        f"  patches: size={config.patch_size} stride={config.patch_stride} max_strokes={config.max_strokes_per_patch}",
        flush=True,
    )
    print(
        f"  patch filtering: include_zero_target_changed_patches={config.include_zero_target_changed_patches}",
        flush=True,
    )
    print(
        f"  training renderer: {config.training_renderer} "
        f"require_structure_targets={config.require_structure_targets} "
        f"require_target_contract={config.require_target_contract}",
        flush=True,
    )

    full_train_dataset = _build_dataset(config, "Train")
    full_val_dataset = _build_dataset(config, "Val")
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset
    if config.overfit_samples:
        indices = _first_sample_indices(full_train_dataset, config.overfit_samples)
        train_dataset = Subset(full_train_dataset, indices)
        val_dataset = Subset(full_train_dataset, indices)
        selected_samples = sorted({full_train_dataset.patch_index[index].sample_id for index in indices})
        print(
            f"  overfit mode: {len(selected_samples)} Train sample(s), {len(indices)} patch(es)",
            flush=True,
        )
    elif config.overfit_patches:
        count = min(config.overfit_patches, len(full_train_dataset))
        indices = list(range(count))
        train_dataset = Subset(full_train_dataset, indices)
        val_dataset = Subset(full_train_dataset, indices)
        print(f"  overfit mode: first {count} Train patch(es)", flush=True)
    if config.train_repeat_factor > 1:
        train_dataset = ConcatDataset([train_dataset] * config.train_repeat_factor)
        print(f"  train repeat factor: {config.train_repeat_factor}", flush=True)
    print(f"  Train patches: {len(train_dataset)}", flush=True)
    print(f"  Val patches: {len(val_dataset)}", flush=True)

    train_loader = _build_loader(
        train_dataset,
        config.batch_size,
        shuffle=not bool(config.overfit_patches),
        num_workers=num_workers,
        device=device,
    )
    val_loader = _build_loader(val_dataset, config.batch_size, shuffle=False, num_workers=num_workers, device=device)
    model_config = VisualDeltaStrokeCompilerConfig(
        model_dim=config.model_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.decoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        grid_size=config.grid_size,
        max_strokes=config.max_strokes_per_patch,
        **_slot_layout_for_max_strokes(config.max_strokes_per_patch),
    )
    model = VisualDeltaStrokeCompiler(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_visual_improvement_rate = -1.0
    for epoch in range(1, config.epochs + 1):
        print(f"epoch {epoch}/{config.epochs} train start", flush=True)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            phase="train",
        )
        print(f"epoch {epoch}/{config.epochs} val start", flush=True)
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            config=config,
            epoch=epoch,
            phase="val",
        )
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics.append(epoch_metrics)
        print(
            f"epoch {epoch}/{config.epochs} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}",
            flush=True,
        )
        _save_checkpoint(
            config.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_metrics,
            metrics_log=metrics,
            model_config=model_config,
            dataset_config=config,
            train_dataset=train_dataset,
            checkpoint_type="epoch",
        )
        print(f"  wrote {config.output_dir / 'latest.pt'}", flush=True)
        visual_metrics = _run_visual_validation(config, epoch)
        if visual_metrics is not None:
            epoch_metrics["visual"] = visual_metrics
            _write_json(config.output_dir / "visual_metrics.json", _visual_metrics_log(metrics))
            print(
                f"  visual validation: improvement_rate={visual_metrics['visual_improvement_rate']:.3f} "
                f"low_change_rate={visual_metrics['low_change_rate']:.3f} "
                f"selected_count={visual_metrics['selected_count']:.0f} "
                f"max_present={visual_metrics['max_present']:.6f}",
                flush=True,
            )
        is_better, best_val_loss, best_visual_improvement_rate = _is_better_visual_delta_checkpoint(
            val_loss=val_metrics["loss"],
            visual_metrics=visual_metrics,
            best_val_loss=best_val_loss,
            best_visual_improvement_rate=best_visual_improvement_rate,
            require_visual_gate=True,
        )
        if is_better:
            _save_checkpoint(
                config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                metrics_log=metrics,
                model_config=model_config,
                dataset_config=config,
                train_dataset=train_dataset,
                checkpoint_type="best",
            )
            print(f"  wrote {config.output_dir / 'best.pt'}", flush=True)
        _write_json(config.output_dir / "metrics.json", metrics)
        print(f"  wrote {config.output_dir / 'metrics.json'}", flush=True)
        if _should_early_stop_zero_selected(config, visual_metrics, epoch):
            print(
                f"  early stop: selected_count stayed zero through epoch {epoch}",
                flush=True,
            )
            break

    return {
        "epochs": metrics,
        "best_val_loss": best_val_loss,
        "output_dir": str(config.output_dir),
        "train_patch_count": len(train_dataset),
        "val_patch_count": len(val_dataset),
    }


def _build_dataset(config: VisualDeltaTrainingConfig, split: str) -> VisualDeltaStrokeDataset:
    return VisualDeltaStrokeDataset(
        config.data_root / split,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        max_strokes_per_patch=config.max_strokes_per_patch,
        mask_threshold=config.mask_threshold,
        min_changed_pixels=config.min_changed_pixels,
        negative_patch_ratio=config.negative_patch_ratio,
        edge_focused_sampling=config.edge_focused_sampling,
        include_zero_target_changed_patches=config.include_zero_target_changed_patches,
        require_structure_targets=config.require_structure_targets,
        require_target_contract=config.require_target_contract,
    )


def _run_epoch(
    model: VisualDeltaStrokeCompiler,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: VisualDeltaTrainingConfig,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "numeric": 0.0,
        "brush": 0.0,
        "present": 0.0,
        "count": 0.0,
        "anti_dot": 0.0,
        "color_clamp": 0.0,
        "size_distribution": 0.0,
        "image": 0.0,
        "preservation": 0.0,
        "gradient": 0.0,
        "edge": 0.0,
        "low_frequency": 0.0,
        "recall": 0.0,
    }
    total_valid = 0
    batches = 0
    loader_length = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        batch = visual_delta_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            loss = compute_visual_delta_loss(
                prediction,
                batch,
                numeric_weight=config.numeric_weight,
                brush_weight=config.brush_weight,
                present_weight=config.present_weight,
                present_positive_weight=config.present_positive_weight,
                count_weight=config.count_weight,
                image_weight=config.image_weight,
                preservation_weight=config.preservation_weight,
                gradient_weight=config.gradient_weight * config.coarse_structure_weight,
                edge_weight=config.edge_weight * config.coarse_structure_weight,
                low_frequency_weight=config.low_frequency_weight,
                recall_weight=config.recall_weight,
                anti_dot_weight=config.anti_dot_weight,
                color_clamp_weight=config.color_clamp_weight,
                size_distribution_weight=config.size_distribution_weight,
                slot_aware_targets=config.slot_aware_targets,
                training_renderer=config.training_renderer,
            )
            if training:
                loss.total.backward()
                optimizer.step()
        weight = max(loss.valid_target_count, 1)
        totals["loss"] += float(loss.total.detach().cpu()) * weight
        totals["numeric"] += float(loss.numeric.detach().cpu()) * weight
        totals["brush"] += float(loss.brush.detach().cpu()) * weight
        totals["present"] += float(loss.present.detach().cpu()) * weight
        totals["count"] += float(loss.count.detach().cpu()) * weight
        totals["anti_dot"] += float(loss.anti_dot.detach().cpu()) * weight
        totals["color_clamp"] += float(loss.color_clamp.detach().cpu()) * weight
        totals["size_distribution"] += float(loss.size_distribution.detach().cpu()) * weight
        totals["image"] += float(loss.image.detach().cpu()) * weight
        totals["preservation"] += float(loss.preservation.detach().cpu()) * weight
        totals["gradient"] += float(loss.gradient.detach().cpu()) * weight
        totals["edge"] += float(loss.edge.detach().cpu()) * weight
        totals["low_frequency"] += float(loss.low_frequency.detach().cpu()) * weight
        totals["recall"] += float(loss.recall.detach().cpu()) * weight
        total_valid += loss.valid_target_count
        batches += 1
        if _should_log_batch(batch_index, loader_length, config.log_every):
            denominator = max(total_valid, 1)
            print(
                f"epoch {epoch} {phase} batch {batch_index}/{loader_length} "
                f"loss={totals['loss'] / denominator:.6f} "
                f"numeric={totals['numeric'] / denominator:.6f} "
                f"brush={totals['brush'] / denominator:.6f} "
                f"present={totals['present'] / denominator:.6f} "
                f"count={totals['count'] / denominator:.6f} "
                f"anti_dot={totals['anti_dot'] / denominator:.6f} "
                f"color_clamp={totals['color_clamp'] / denominator:.6f} "
                f"size_dist={totals['size_distribution'] / denominator:.6f} "
                f"image={totals['image'] / denominator:.6f} "
                f"preserve={totals['preservation'] / denominator:.6f} "
                f"gradient={totals['gradient'] / denominator:.6f} "
                f"edge={totals['edge'] / denominator:.6f} "
                f"low_freq={totals['low_frequency'] / denominator:.6f} "
                f"recall={totals['recall'] / denominator:.6f} "
                f"valid={total_valid}",
                flush=True,
            )
    denominator = max(total_valid, 1)
    return {
        "loss": totals["loss"] / denominator,
        "numeric_loss": totals["numeric"] / denominator,
        "brush_loss": totals["brush"] / denominator,
        "present_loss": totals["present"] / denominator,
        "count_loss": totals["count"] / denominator,
        "anti_dot_loss": totals["anti_dot"] / denominator,
        "color_clamp_loss": totals["color_clamp"] / denominator,
        "size_distribution_loss": totals["size_distribution"] / denominator,
        "image_loss": totals["image"] / denominator,
        "preservation_loss": totals["preservation"] / denominator,
        "gradient_loss": totals["gradient"] / denominator,
        "edge_loss": totals["edge"] / denominator,
        "low_frequency_loss": totals["low_frequency"] / denominator,
        "recall_loss": totals["recall"] / denominator,
        "valid_target_count": float(total_valid),
        "batches": float(batches),
    }


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_visual_delta_patches,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _slot_layout_for_max_strokes(max_strokes: int) -> dict[str, int]:
    if max_strokes <= 0:
        raise ValueError("visual-delta compiler needs at least 1 stroke proposal")
    if max_strokes <= 4:
        return {"coarse_grid_size": 1, "detail_grid_rows": 1, "detail_grid_cols": max(1, max_strokes - 1)}
    coarse_grid_size = max(1, int((max_strokes * 0.25) ** 0.5))
    rows = int(max_strokes**0.5)
    while rows > 1 and max_strokes % rows != 0:
        rows -= 1
    cols = max_strokes // rows
    return {"coarse_grid_size": coarse_grid_size, "detail_grid_rows": rows, "detail_grid_cols": cols}


def _save_checkpoint(
    path: Path,
    model: VisualDeltaStrokeCompiler,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    metrics_log: list[dict[str, Any]],
    model_config: VisualDeltaStrokeCompilerConfig,
    dataset_config: VisualDeltaTrainingConfig,
    train_dataset,
    checkpoint_type: str,
) -> None:
    tokenizer = _base_dataset(train_dataset).tokenizer
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "metrics_log": metrics_log,
            "model_config": asdict(model_config),
            "dataset_config": {
                "patch_size": dataset_config.patch_size,
                "patch_stride": dataset_config.patch_stride,
                "max_strokes_per_patch": dataset_config.max_strokes_per_patch,
                "mask_threshold": dataset_config.mask_threshold,
                "min_changed_pixels": dataset_config.min_changed_pixels,
                "negative_patch_ratio": dataset_config.negative_patch_ratio,
                "edge_focused_sampling": dataset_config.edge_focused_sampling,
                "include_zero_target_changed_patches": dataset_config.include_zero_target_changed_patches,
                "count_weight": dataset_config.count_weight,
                "image_weight": dataset_config.image_weight,
                "preservation_weight": dataset_config.preservation_weight,
                "gradient_weight": dataset_config.gradient_weight,
                "edge_weight": dataset_config.edge_weight,
                "low_frequency_weight": dataset_config.low_frequency_weight,
                "coarse_structure_weight": dataset_config.coarse_structure_weight,
                "recall_weight": dataset_config.recall_weight,
                "anti_dot_weight": dataset_config.anti_dot_weight,
                "color_clamp_weight": dataset_config.color_clamp_weight,
                "size_distribution_weight": dataset_config.size_distribution_weight,
                "slot_aware_targets": dataset_config.slot_aware_targets,
                "training_renderer": dataset_config.training_renderer,
                "present_threshold": dataset_config.present_threshold,
                "min_export_candidates_per_sample": dataset_config.min_export_candidates_per_sample,
                "present_positive_weight": dataset_config.present_positive_weight,
                "max_export_strokes_per_sample": dataset_config.max_export_strokes_per_sample,
                "max_export_strokes_per_patch": dataset_config.max_export_strokes_per_patch,
                "min_export_render_area": dataset_config.min_export_render_area,
                "export_ranking_mode": dataset_config.export_ranking_mode,
                "target_vram_gb": dataset_config.target_vram_gb,
                "require_structure_targets": dataset_config.require_structure_targets,
                "require_target_contract": dataset_config.require_target_contract,
                "early_stop_zero_selected_epochs": dataset_config.early_stop_zero_selected_epochs,
            },
            "checkpoint_type": checkpoint_type,
            "tokenizer": {
                "brush_to_id": tokenizer.brush_to_id,
                "id_to_brush": tokenizer.id_to_brush,
                "max_strokes": tokenizer.max_strokes,
                "numeric_dim": tokenizer.numeric_dim,
            },
        },
        path,
    )


def _run_visual_validation(config: VisualDeltaTrainingConfig, epoch: int) -> dict[str, Any] | None:
    if config.visual_validation_samples <= 0:
        return None
    if epoch % config.visual_validation_interval != 0 and epoch != config.epochs:
        return None
    from Source.Model.export_visual_delta_predictions import ExportVisualDeltaConfig, export_visual_delta_predictions

    split = "Train" if config.overfit_patches or config.overfit_samples else "Val"
    output_root = config.output_dir / "VisualValidation" / f"epoch_{epoch:04d}"
    exported = export_visual_delta_predictions(
        ExportVisualDeltaConfig(
            data_root=config.data_root,
            checkpoint=config.output_dir / "latest.pt",
            output_root=output_root,
            split=split,
            limit=config.visual_validation_samples,
            device=config.visual_validation_device,
            cuda_attention_backend=config.cuda_attention_backend,
            min_changed_pixel_ratio=config.min_visual_changed_pixel_ratio,
            min_gradient_improvement=config.min_visual_gradient_improvement,
            min_edge_overlap=config.min_visual_edge_overlap,
            present_threshold=config.present_threshold,
            min_export_candidates_per_sample=config.min_export_candidates_per_sample,
            max_strokes_per_sample=config.max_export_strokes_per_sample,
            max_strokes_per_patch=config.max_export_strokes_per_patch,
            min_render_area=config.min_export_render_area,
            ranking_mode=config.export_ranking_mode,
            allow_visual_failed_checkpoint=True,
        )
    )
    rendered = [entry for entry in exported if entry.get("visual_improved") is not None]
    improved = [entry for entry in rendered if entry.get("visual_improved")]
    low_change = [entry for entry in rendered if entry.get("status") == "failed_low_pixel_change"]
    structure_failed = [entry for entry in rendered if entry.get("status") == "failed_structure_noise"]
    structure_metrics = _average_structure_metrics(rendered)
    export_filter_metrics = _average_export_filter_metrics(exported)
    visual_pass = bool(improved) and not low_change and not structure_failed
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
        "structure_failed_count": len(structure_failed),
        "structure_failed_rate": len(structure_failed) / len(rendered) if rendered else 0.0,
        **structure_metrics,
        **export_filter_metrics,
        "checkpoint_status": "visual_pass" if visual_pass else "visual_failed",
        "status_histogram": _status_histogram(exported),
    }
    _write_json(output_root / "visual_summary.json", summary)
    return summary


def _base_dataset(dataset):
    if isinstance(dataset, Subset):
        return _base_dataset(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return _base_dataset(dataset.datasets[0])
    return dataset


def _visual_metrics_log(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"epoch": entry["epoch"], "visual": entry["visual"]}
        for entry in metrics
        if isinstance(entry, dict) and isinstance(entry.get("visual"), dict)
    ]


def _validate_config(config: VisualDeltaTrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.numeric_weight < 0.0:
        raise ValueError("numeric_weight must be non-negative")
    if config.brush_weight < 0.0:
        raise ValueError("brush_weight must be non-negative")
    if config.present_weight < 0.0:
        raise ValueError("present_weight must be non-negative")
    if config.present_positive_weight <= 0.0:
        raise ValueError("present_positive_weight must be positive")
    if config.count_weight < 0.0:
        raise ValueError("count_weight must be non-negative")
    if config.image_weight < 0.0:
        raise ValueError("image_weight must be non-negative")
    if config.preservation_weight < 0.0:
        raise ValueError("preservation_weight must be non-negative")
    if config.gradient_weight < 0.0:
        raise ValueError("gradient_weight must be non-negative")
    if config.edge_weight < 0.0:
        raise ValueError("edge_weight must be non-negative")
    if config.low_frequency_weight < 0.0:
        raise ValueError("low_frequency_weight must be non-negative")
    if config.coarse_structure_weight < 0.0:
        raise ValueError("coarse_structure_weight must be non-negative")
    if config.recall_weight < 0.0:
        raise ValueError("recall_weight must be non-negative")
    if config.anti_dot_weight < 0.0:
        raise ValueError("anti_dot_weight must be non-negative")
    if config.color_clamp_weight < 0.0:
        raise ValueError("color_clamp_weight must be non-negative")
    if config.size_distribution_weight < 0.0:
        raise ValueError("size_distribution_weight must be non-negative")
    if config.training_renderer not in SUPPORTED_TRAINING_RENDERERS:
        raise ValueError(
            f"training_renderer must be one of {', '.join(SUPPORTED_TRAINING_RENDERERS)}"
        )
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.overfit_patches < 0:
        raise ValueError("overfit_patches must be non-negative")
    if config.overfit_samples < 0:
        raise ValueError("overfit_samples must be non-negative")
    if config.overfit_patches and config.overfit_samples:
        raise ValueError("use either overfit_patches or overfit_samples, not both")
    if config.train_repeat_factor <= 0:
        raise ValueError("train_repeat_factor must be positive")
    if config.visual_validation_interval <= 0:
        raise ValueError("visual_validation_interval must be positive")
    if not 0.0 <= config.present_threshold <= 1.0:
        raise ValueError("present_threshold must be in [0, 1]")
    if config.max_export_strokes_per_sample <= 0:
        raise ValueError("max_export_strokes_per_sample must be positive")
    if config.max_export_strokes_per_patch <= 0:
        raise ValueError("max_export_strokes_per_patch must be positive")
    if config.min_export_render_area < 0.0:
        raise ValueError("min_export_render_area must be non-negative")
    if config.min_export_candidates_per_sample < 0:
        raise ValueError("min_export_candidates_per_sample must be non-negative")
    if config.early_stop_zero_selected_epochs < 0:
        raise ValueError("early_stop_zero_selected_epochs must be non-negative")
    if config.export_ranking_mode != "visual-delta":
        raise ValueError("export_ranking_mode must be 'visual-delta'")
    if config.target_vram_gb <= 0:
        raise ValueError("target_vram_gb must be positive")
    if config.min_visual_edge_overlap < 0.0:
        raise ValueError("min_visual_edge_overlap must be non-negative")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BrushWright visual-delta stroke compiler.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--model-dim", type=int, default=DEFAULT_MODEL_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--decoder-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_FF_DIM)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--patch-stride", type=int, default=DEFAULT_PATCH_STRIDE)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--max-strokes-per-patch", type=int, default=DEFAULT_MAX_STROKES)
    parser.add_argument("--mask-threshold", type=float, default=0.04)
    parser.add_argument("--min-changed-pixels", type=int, default=16)
    parser.add_argument("--negative-patch-ratio", type=float, default=0.25)
    parser.add_argument("--no-edge-focused-sampling", action="store_true")
    parser.add_argument("--include-zero-target-changed-patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overfit-patches", type=int, default=0)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--train-repeat-factor", type=int, default=1)
    parser.add_argument("--numeric-weight", type=float, default=1.0)
    parser.add_argument("--brush-weight", type=float, default=0.25)
    parser.add_argument("--present-weight", type=float, default=1.0)
    parser.add_argument("--present-positive-weight", type=float, default=DEFAULT_PRESENT_POSITIVE_WEIGHT)
    parser.add_argument("--count-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=4.0)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=2.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--low-frequency-weight", type=float, default=1.0)
    parser.add_argument("--coarse-structure-weight", type=float, default=2.0)
    parser.add_argument("--recall-weight", type=float, default=0.5)
    parser.add_argument("--anti-dot-weight", type=float, default=0.75)
    parser.add_argument("--color-clamp-weight", type=float, default=1.0)
    parser.add_argument("--size-distribution-weight", type=float, default=0.5)
    parser.add_argument("--slot-aware-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--training-renderer",
        choices=SUPPORTED_TRAINING_RENDERERS,
        default=TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
    )
    parser.add_argument("--require-structure-targets", action="store_true")
    parser.add_argument("--require-target-contract", default=TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET)
    parser.add_argument("--visual-validation-samples", type=int, default=8)
    parser.add_argument("--visual-validation-device", default="cpu")
    parser.add_argument("--visual-validation-interval", type=int, default=1)
    parser.add_argument("--min-visual-changed-pixel-ratio", type=float, default=0.005)
    parser.add_argument("--min-visual-gradient-improvement", type=float, default=0.0)
    parser.add_argument("--min-visual-edge-overlap", type=float, default=0.02)
    parser.add_argument("--present-threshold", type=float, default=DEFAULT_PRESENT_THRESHOLD)
    parser.add_argument("--min-export-candidates-per-sample", type=int, default=0)
    parser.add_argument("--max-export-strokes-per-sample", type=int, default=DEFAULT_MAX_STROKES)
    parser.add_argument("--max-export-strokes-per-patch", type=int, default=DEFAULT_MAX_STROKES)
    parser.add_argument("--min-export-render-area", type=float, default=8.0)
    parser.add_argument("--export-ranking-mode", choices=("visual-delta",), default="visual-delta")
    parser.add_argument("--target-vram-gb", type=int, default=DEFAULT_TARGET_VRAM_GB)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cuda-attention-backend", choices=("math", "default"), default=DEFAULT_CUDA_ATTENTION_BACKEND)
    parser.add_argument("--early-stop-zero-selected-epochs", type=int, default=3)
    return parser


def _first_sample_indices(dataset: VisualDeltaStrokeDataset, sample_count: int) -> list[int]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    sample_ids: list[str] = []
    for sample_entry in dataset.manifest.get("samples", []):
        sample_id = str(sample_entry["sample_id"])
        if sample_id not in sample_ids:
            sample_ids.append(sample_id)
        if len(sample_ids) >= sample_count:
            break
    selected = set(sample_ids)
    indices = [index for index, patch in enumerate(dataset.patch_index) if patch.sample_id in selected]
    if not indices:
        raise ValueError(f"no patches found for first {sample_count} sample(s)")
    return indices


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_better_visual_delta_checkpoint(
    val_loss: float,
    visual_metrics: dict[str, Any] | None,
    best_val_loss: float,
    best_visual_improvement_rate: float,
    require_visual_gate: bool = True,
) -> tuple[bool, float, float]:
    if visual_metrics is None:
        if require_visual_gate:
            return False, best_val_loss, best_visual_improvement_rate
        return val_loss < best_val_loss, min(best_val_loss, val_loss), best_visual_improvement_rate

    visual_rate = float(visual_metrics.get("visual_improvement_rate", 0.0))
    structure_failed_rate = float(visual_metrics.get("structure_failed_rate", 1.0))
    low_change_rate = float(visual_metrics.get("low_change_rate", 1.0))
    visual_gate_passed = (
        visual_metrics.get("checkpoint_status") == "visual_pass"
        and visual_rate > 0.0
        and structure_failed_rate == 0.0
        and low_change_rate == 0.0
    )
    if not visual_gate_passed:
        return False, best_val_loss, best_visual_improvement_rate
    if visual_rate > best_visual_improvement_rate:
        return True, val_loss, visual_rate
    if visual_rate == best_visual_improvement_rate and val_loss < best_val_loss:
        return True, val_loss, visual_rate
    return False, best_val_loss, best_visual_improvement_rate


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


def _should_early_stop_zero_selected(
    config: VisualDeltaTrainingConfig,
    visual_metrics: dict[str, Any] | None,
    epoch: int,
) -> bool:
    if config.early_stop_zero_selected_epochs <= 0:
        return False
    if epoch < config.early_stop_zero_selected_epochs:
        return False
    if visual_metrics is None:
        return False
    return float(visual_metrics.get("selected_count", 0.0)) == 0.0


if __name__ == "__main__":
    raise SystemExit(main())
