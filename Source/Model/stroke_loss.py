"""Loss functions for BrushWright stroke chunk prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from Source.Model.stroke_dataset import StrokeBatch
from Source.Model.stroke_decoder import StrokePredictionOutput
from Source.Model.paint_transformer_soft_renderer import render_paint_transformer_soft_strokes
from Source.Model.stroke_tokenizer import NUMERIC_FIELDS


DEFAULT_NUMERIC_FIELD_WEIGHTS = torch.tensor(
    [4.0, 4.0, 2.0, 4.0, 4.0, 0.25, 3.0, 3.0, 3.0],
    dtype=torch.float32,
)
DEFAULT_DISTRIBUTION_FIELD_WEIGHTS = torch.tensor(
    [4.0, 4.0, 1.0, 4.0, 4.0, 0.0, 3.0, 3.0, 3.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class StrokeLossOutput:
    total: torch.Tensor
    numeric: torch.Tensor
    brush: torch.Tensor
    distribution: torch.Tensor
    visual: torch.Tensor
    valid_target_count: int


def compute_stroke_loss(
    prediction: StrokePredictionOutput,
    batch: StrokeBatch,
    numeric_weight: float = 1.0,
    brush_weight: float = 0.25,
    distribution_weight: float = 0.2,
    visual_weight: float = 1.0,
    set_matching: bool = True,
    numeric_field_weights: torch.Tensor | None = None,
    distribution_field_weights: torch.Tensor | None = None,
) -> StrokeLossOutput:
    if prediction.pred_numeric.shape != batch.target_numeric.shape:
        raise ValueError("pred_numeric and target_numeric shapes must match")
    if prediction.pred_brush_logits.shape[:2] != batch.target_brush_ids.shape:
        raise ValueError("pred_brush_logits and target_brush_ids batch/sequence shapes must match")
    if batch.target_padding_mask.shape != batch.target_brush_ids.shape:
        raise ValueError("target_padding_mask and target_brush_ids shapes must match")

    target_numeric = batch.target_numeric
    target_brush_ids = batch.target_brush_ids
    target_padding_mask = batch.target_padding_mask
    if set_matching:
        target_numeric, target_brush_ids, target_padding_mask = match_stroke_targets(
            prediction.pred_numeric,
            batch.target_numeric,
            batch.target_brush_ids,
            batch.target_padding_mask,
            field_weights=numeric_field_weights,
        )

    valid_mask = ~target_padding_mask
    valid_count_tensor = valid_mask.sum()
    valid_count = int(valid_count_tensor.item())
    if valid_count == 0:
        zero = prediction.pred_numeric.sum() * 0.0
        return StrokeLossOutput(total=zero, numeric=zero, brush=zero, distribution=zero, visual=zero, valid_target_count=0)

    field_weights = _field_weights(
        numeric_field_weights,
        default=DEFAULT_NUMERIC_FIELD_WEIGHTS,
        device=prediction.pred_numeric.device,
        dtype=prediction.pred_numeric.dtype,
    )
    numeric_loss = F.smooth_l1_loss(
        prediction.pred_numeric,
        target_numeric,
        reduction="none",
    )
    numeric_loss = (numeric_loss * valid_mask.unsqueeze(-1) * field_weights).sum() / (
        valid_count_tensor.to(numeric_loss.dtype) * field_weights.sum().clamp_min(1.0)
    )

    brush_loss = F.cross_entropy(
        prediction.pred_brush_logits.reshape(-1, prediction.pred_brush_logits.shape[-1]),
        target_brush_ids.reshape(-1),
        reduction="none",
    )
    brush_loss = (brush_loss * valid_mask.reshape(-1).to(brush_loss.dtype)).sum() / valid_count_tensor.to(
        brush_loss.dtype
    )

    distribution_loss = compute_distribution_loss(
        prediction.pred_numeric,
        target_numeric,
        target_padding_mask,
        field_weights=distribution_field_weights,
    )
    visual_loss = compute_render_loss(
        prediction.pred_numeric,
        batch,
        target_padding_mask=target_padding_mask,
    )
    total = (
        numeric_loss * numeric_weight
        + brush_loss * brush_weight
        + distribution_loss * distribution_weight
        + visual_loss * visual_weight
    )
    return StrokeLossOutput(
        total=total,
        numeric=numeric_loss,
        brush=brush_loss,
        distribution=distribution_loss,
        visual=visual_loss,
        valid_target_count=valid_count,
    )


def match_stroke_targets(
    pred_numeric: torch.Tensor,
    target_numeric: torch.Tensor,
    target_brush_ids: torch.Tensor,
    target_padding_mask: torch.Tensor,
    field_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign target strokes to prediction slots by weighted numeric proximity.

    This removes the accidental requirement that target stroke 0 must be decoded
    by slot 0. The assignment is deterministic greedy matching over at most 64
    target strokes, which is enough for the V1 chunk size without adding a new
    dependency.
    """

    if pred_numeric.shape != target_numeric.shape:
        raise ValueError("pred_numeric and target_numeric shapes must match")
    if target_brush_ids.shape != target_padding_mask.shape:
        raise ValueError("target_brush_ids and target_padding_mask shapes must match")
    if target_brush_ids.shape != target_numeric.shape[:2]:
        raise ValueError("target_brush_ids must have shape [batch, seq]")

    weights = _field_weights(
        field_weights,
        default=DEFAULT_NUMERIC_FIELD_WEIGHTS,
        device=pred_numeric.device,
        dtype=pred_numeric.dtype,
    )
    matched_numeric = torch.zeros_like(target_numeric)
    matched_brush_ids = torch.zeros_like(target_brush_ids)
    matched_padding_mask = torch.ones_like(target_padding_mask, dtype=torch.bool)

    batch_size, seq_len, _ = pred_numeric.shape
    for batch_index in range(batch_size):
        valid_target_indices = torch.nonzero(~target_padding_mask[batch_index], as_tuple=False).flatten()
        valid_target_count = int(valid_target_indices.numel())
        if valid_target_count == 0:
            continue
        valid_targets = target_numeric[batch_index, valid_target_indices]
        costs = (
            torch.abs(pred_numeric[batch_index].detach().unsqueeze(1) - valid_targets.detach().unsqueeze(0))
            * weights.view(1, 1, -1)
        ).sum(dim=-1)
        flat_order = torch.argsort(costs.reshape(-1), stable=True)
        used_pred: set[int] = set()
        used_target: set[int] = set()
        for flat_index_tensor in flat_order:
            flat_index = int(flat_index_tensor.item())
            pred_index = flat_index // valid_target_count
            target_list_index = flat_index % valid_target_count
            if pred_index in used_pred or target_list_index in used_target:
                continue
            target_index = int(valid_target_indices[target_list_index].item())
            matched_numeric[batch_index, pred_index] = target_numeric[batch_index, target_index]
            matched_brush_ids[batch_index, pred_index] = target_brush_ids[batch_index, target_index]
            matched_padding_mask[batch_index, pred_index] = False
            used_pred.add(pred_index)
            used_target.add(target_list_index)
            if len(used_target) == valid_target_count or len(used_pred) == seq_len:
                break

    return matched_numeric, matched_brush_ids, matched_padding_mask


def compute_render_loss(
    pred_numeric: torch.Tensor,
    batch: StrokeBatch,
    target_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render predicted strokes over the draft and compare against the finished image."""

    if batch.draft_images is None:
        raise ValueError("draft_images are required for render loss")
    if batch.goal_images is None:
        raise ValueError("goal_images are required for render loss")
    if pred_numeric.shape[:2] != batch.target_padding_mask.shape:
        raise ValueError("pred_numeric and target_padding_mask batch/sequence shapes must match")
    padding_mask = batch.target_padding_mask if target_padding_mask is None else target_padding_mask
    if padding_mask.shape != batch.target_padding_mask.shape:
        raise ValueError("target_padding_mask must match the batch target padding mask shape")

    valid_logits = torch.where(
        padding_mask,
        torch.full_like(batch.target_padding_mask, -20.0, dtype=pred_numeric.dtype, device=pred_numeric.device),
        torch.full_like(batch.target_padding_mask, 20.0, dtype=pred_numeric.dtype, device=pred_numeric.device),
    )
    rendered = render_paint_transformer_soft_strokes(
        batch.draft_images,
        pred_numeric,
        valid_logits,
    )
    pixel_l1 = torch.abs(rendered - batch.goal_images).mean(dim=1, keepdim=True)
    full_canvas_loss = pixel_l1.mean()
    if batch.error_maps is None:
        return full_canvas_loss

    changed_mask = batch.error_maps.abs().amax(dim=1, keepdim=True) > (1.0 / 255.0)
    changed_weight = changed_mask.to(pixel_l1.dtype)
    changed_count = changed_weight.sum()
    if float(changed_count.detach().cpu()) <= 0.0:
        return full_canvas_loss
    changed_region_loss = (pixel_l1 * changed_weight).sum() / changed_count.clamp_min(1.0)

    outside_weight = 1.0 - changed_weight
    outside_count = outside_weight.sum()
    if float(outside_count.detach().cpu()) > 0.0:
        outside_l1 = torch.abs(rendered - batch.draft_images).mean(dim=1, keepdim=True)
        preservation_loss = (outside_l1 * outside_weight).sum() / outside_count.clamp_min(1.0)
    else:
        preservation_loss = full_canvas_loss * 0.0
    return changed_region_loss + full_canvas_loss * 0.25 + preservation_loss * 0.25


def compute_distribution_loss(
    pred_numeric: torch.Tensor,
    target_numeric: torch.Tensor,
    target_padding_mask: torch.Tensor,
    field_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize collapsed chunk statistics while respecting padded target slots."""

    if pred_numeric.shape != target_numeric.shape:
        raise ValueError("pred_numeric and target_numeric shapes must match")
    if target_padding_mask.shape != target_numeric.shape[:2]:
        raise ValueError("target_padding_mask must have shape [batch, seq]")
    if pred_numeric.shape[-1] != len(NUMERIC_FIELDS):
        raise ValueError(f"numeric tensors must have last dimension {len(NUMERIC_FIELDS)}")

    weights = _field_weights(
        field_weights,
        default=DEFAULT_DISTRIBUTION_FIELD_WEIGHTS,
        device=pred_numeric.device,
        dtype=pred_numeric.dtype,
    )
    valid = (~target_padding_mask).unsqueeze(-1).to(pred_numeric.dtype)
    counts = valid.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred_numeric * valid).sum(dim=1) / counts
    target_mean = (target_numeric * valid).sum(dim=1) / counts

    pred_var = (((pred_numeric - pred_mean.unsqueeze(1)) * valid) ** 2).sum(dim=1) / counts
    target_var = (((target_numeric - target_mean.unsqueeze(1)) * valid) ** 2).sum(dim=1) / counts
    pred_std = torch.sqrt(pred_var.clamp_min(1e-8))
    target_std = torch.sqrt(target_var.clamp_min(1e-8))

    mean_loss = F.smooth_l1_loss(pred_mean, target_mean, reduction="none")
    std_loss = F.smooth_l1_loss(pred_std, target_std, reduction="none")
    weighted = (mean_loss + std_loss) * weights
    return weighted.sum() / (pred_numeric.shape[0] * weights.sum().clamp_min(1.0))


def _field_weights(
    values: torch.Tensor | None,
    default: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = default if values is None else values
    if weights.shape != (len(NUMERIC_FIELDS),):
        raise ValueError(f"field weights must have shape [{len(NUMERIC_FIELDS)}]")
    return weights.to(device=device, dtype=dtype)
