"""Loss functions for the visual-delta stroke compiler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from Source.Model.stroke_loss import DEFAULT_NUMERIC_FIELD_WEIGHTS, _field_weights
from Source.Model.paint_transformer_soft_renderer import render_paint_transformer_soft_strokes
from Source.Model.visual_delta_dataset import VisualDeltaBatch
from Source.Model.visual_delta_predictor import VisualDeltaPredictionOutput


TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT = "paint-transformer-soft"
TRAINING_RENDERER_SOFT_ELLIPSE = "soft-ellipse"
SUPPORTED_TRAINING_RENDERERS = (TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT, TRAINING_RENDERER_SOFT_ELLIPSE)

DEFAULT_VISUAL_DELTA_MATCH_WEIGHTS = torch.tensor(
    [5.0, 5.0, 1.0, 3.0, 3.0, 0.0, 2.0, 2.0, 2.0],
    dtype=torch.float32,
)
DEFAULT_VISUAL_DELTA_NUMERIC_WEIGHTS = torch.tensor(
    [6.0, 6.0, 1.5, 6.0, 6.0, 0.25, 4.0, 4.0, 4.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class VisualDeltaLossOutput:
    total: torch.Tensor
    numeric: torch.Tensor
    brush: torch.Tensor
    present: torch.Tensor
    count: torch.Tensor
    anti_dot: torch.Tensor
    color_clamp: torch.Tensor
    size_distribution: torch.Tensor
    image: torch.Tensor
    preservation: torch.Tensor
    gradient: torch.Tensor
    edge: torch.Tensor
    low_frequency: torch.Tensor
    recall: torch.Tensor
    valid_target_count: int


def compute_visual_delta_loss(
    prediction: VisualDeltaPredictionOutput,
    batch: VisualDeltaBatch,
    numeric_weight: float = 1.0,
    brush_weight: float = 0.25,
    present_weight: float = 1.0,
    present_positive_weight: float = 8.0,
    count_weight: float = 0.5,
    image_weight: float = 4.0,
    preservation_weight: float = 1.0,
    gradient_weight: float = 2.0,
    edge_weight: float = 1.0,
    low_frequency_weight: float = 1.0,
    recall_weight: float = 0.5,
    anti_dot_weight: float = 0.0,
    color_clamp_weight: float = 0.0,
    size_distribution_weight: float = 0.0,
    slot_aware_targets: bool = True,
    numeric_field_weights: torch.Tensor | None = None,
    match_field_weights: torch.Tensor | None = None,
    training_renderer: str = TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
) -> VisualDeltaLossOutput:
    if prediction.pred_numeric.shape != batch.target_numeric.shape:
        raise ValueError("pred_numeric and target_numeric shapes must match")
    if prediction.pred_brush_logits.shape[:2] != batch.target_brush_ids.shape:
        raise ValueError("pred_brush_logits and target_brush_ids batch/sequence shapes must match")
    if prediction.pred_present_logits.shape != batch.target_present.shape:
        raise ValueError("pred_present_logits and target_present shapes must match")

    valid_mask = ~batch.target_padding_mask
    valid_count_tensor = valid_mask.sum()
    valid_count = int(valid_count_tensor.item())
    field_weights = _field_weights(
        numeric_field_weights,
        default=DEFAULT_VISUAL_DELTA_NUMERIC_WEIGHTS,
        device=prediction.pred_numeric.device,
        dtype=prediction.pred_numeric.dtype,
    )
    match_weights = _field_weights(
        match_field_weights,
        default=DEFAULT_VISUAL_DELTA_MATCH_WEIGHTS,
        device=prediction.pred_numeric.device,
        dtype=prediction.pred_numeric.dtype,
    )
    if slot_aware_targets:
        matched_numeric, matched_brush_ids, matched_mask = match_visual_delta_strokes_slot_aware(
            batch.target_numeric,
            batch.target_brush_ids,
            batch.target_padding_mask,
            slot_count=prediction.pred_numeric.shape[1],
        )
    else:
        matched_numeric, matched_brush_ids, matched_mask = match_visual_delta_strokes(
            prediction.pred_numeric.detach(),
            batch.target_numeric,
            batch.target_brush_ids,
            batch.target_padding_mask,
            match_weights,
        )
    matched_count_tensor = matched_mask.sum()
    matched_count = int(matched_count_tensor.item())

    if matched_count:
        numeric = F.smooth_l1_loss(prediction.pred_numeric, matched_numeric, reduction="none")
        numeric = (numeric * matched_mask.unsqueeze(-1) * field_weights).sum() / (
            matched_count_tensor.to(numeric.dtype) * field_weights.sum().clamp_min(1.0)
        )
        brush = F.cross_entropy(
            prediction.pred_brush_logits.reshape(-1, prediction.pred_brush_logits.shape[-1]),
            matched_brush_ids.reshape(-1),
            reduction="none",
        )
        brush = (brush * matched_mask.reshape(-1).to(brush.dtype)).sum() / matched_count_tensor.to(brush.dtype)
    else:
        numeric = prediction.pred_numeric.sum() * 0.0
        brush = prediction.pred_brush_logits.sum() * 0.0

    present_target = matched_mask.to(prediction.pred_present_logits.dtype)
    present_loss = F.binary_cross_entropy_with_logits(
        prediction.pred_present_logits,
        present_target,
        reduction="none",
    )
    present_slot_weights = torch.where(
        present_target > 0.0,
        torch.full_like(present_loss, present_positive_weight),
        torch.ones_like(present_loss),
    )
    present = (present_loss * present_slot_weights).sum() / present_slot_weights.sum().clamp_min(1.0)
    count = compute_present_count_loss(
        prediction.pred_present_logits,
        target_present_count=valid_mask.sum(dim=1).to(prediction.pred_present_logits.dtype),
    )
    image, preservation, gradient, edge, low_frequency = compute_visual_patch_loss(
        prediction,
        batch,
        training_renderer=training_renderer,
    )
    anti_dot = compute_anti_dot_loss(prediction.pred_numeric, prediction.pred_present_logits, batch.patch_tensor)
    color_clamp = compute_color_clamp_loss(prediction.pred_numeric, prediction.pred_present_logits, batch.patch_tensor)
    size_distribution = compute_assigned_size_distribution_loss(
        prediction.pred_numeric,
        matched_numeric,
        matched_mask,
    )
    recall = compute_present_recall_loss(
        prediction.pred_present_logits,
        target_present_count=valid_mask.sum(dim=1).to(prediction.pred_present_logits.dtype),
    )
    total = (
        numeric * numeric_weight
        + brush * brush_weight
        + present * present_weight
        + count * count_weight
        + image * image_weight
        + preservation * preservation_weight
        + gradient * gradient_weight
        + edge * edge_weight
        + low_frequency * low_frequency_weight
        + recall * recall_weight
        + anti_dot * anti_dot_weight
        + color_clamp * color_clamp_weight
        + size_distribution * size_distribution_weight
    )
    return VisualDeltaLossOutput(
        total=total,
        numeric=numeric,
        brush=brush,
        present=present,
        count=count,
        anti_dot=anti_dot,
        color_clamp=color_clamp,
        size_distribution=size_distribution,
        image=image,
        preservation=preservation,
        gradient=gradient,
        edge=edge,
        low_frequency=low_frequency,
        recall=recall,
        valid_target_count=valid_count,
    )


def match_visual_delta_strokes_slot_aware(
    target_numeric: torch.Tensor,
    target_brush_ids: torch.Tensor,
    target_padding_mask: torch.Tensor,
    slot_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign target strokes to deterministic coarse/detail slots by nearest anchor."""

    if target_brush_ids.shape != target_numeric.shape[:2]:
        raise ValueError("target_brush_ids must have shape [batch, seq]")
    if target_padding_mask.shape != target_numeric.shape[:2]:
        raise ValueError("target_padding_mask must have shape [batch, seq]")
    batch_size, target_slots, field_count = target_numeric.shape
    resolved_slot_count = target_slots if slot_count is None else slot_count
    if resolved_slot_count != target_slots:
        raise ValueError("slot_count must match target_numeric sequence length")
    anchors, roles = _slot_anchors(resolved_slot_count, target_numeric.device, target_numeric.dtype)
    matched_numeric = torch.zeros_like(target_numeric)
    matched_brush_ids = torch.zeros_like(target_brush_ids)
    matched_mask = torch.zeros_like(target_padding_mask, dtype=torch.bool)
    for batch_index in range(batch_size):
        target_indices = torch.nonzero(~target_padding_mask[batch_index], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        target = target_numeric[batch_index, target_indices]
        target_roles = _target_roles_for_numeric(target)
        order = _target_assignment_order(target)
        used_slots: set[int] = set()
        for local_index in order.tolist():
            target_role = int(target_roles[local_index].item())
            compatible = torch.nonzero(roles == target_role, as_tuple=False).flatten()
            if compatible.numel() == 0:
                compatible = torch.arange(resolved_slot_count, device=target_numeric.device)
            available = [int(slot.item()) for slot in compatible if int(slot.item()) not in used_slots]
            if not available:
                available = [slot for slot in range(resolved_slot_count) if slot not in used_slots]
            if not available:
                break
            available_tensor = torch.tensor(available, device=target_numeric.device, dtype=torch.long)
            xy = target[local_index, 0:2]
            distances = torch.sum(torch.square(anchors[available_tensor] - xy), dim=-1)
            slot = int(available_tensor[int(torch.argmin(distances).item())].item())
            original_target_index = int(target_indices[local_index].item())
            matched_numeric[batch_index, slot] = target_numeric[batch_index, original_target_index]
            matched_brush_ids[batch_index, slot] = target_brush_ids[batch_index, original_target_index]
            matched_mask[batch_index, slot] = True
            used_slots.add(slot)
    return matched_numeric, matched_brush_ids, matched_mask


def match_visual_delta_strokes(
    pred_numeric: torch.Tensor,
    target_numeric: torch.Tensor,
    target_brush_ids: torch.Tensor,
    target_padding_mask: torch.Tensor,
    field_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily match unordered target strokes to prediction slots by weighted numeric distance."""

    if pred_numeric.shape != target_numeric.shape:
        raise ValueError("pred_numeric and target_numeric shapes must match")
    if target_brush_ids.shape != target_numeric.shape[:2]:
        raise ValueError("target_brush_ids must have shape [batch, seq]")
    if target_padding_mask.shape != target_numeric.shape[:2]:
        raise ValueError("target_padding_mask must have shape [batch, seq]")
    weights = _field_weights(
        field_weights,
        default=DEFAULT_VISUAL_DELTA_MATCH_WEIGHTS,
        device=pred_numeric.device,
        dtype=pred_numeric.dtype,
    )
    batch_size, slot_count, field_count = pred_numeric.shape
    matched_numeric = torch.zeros_like(target_numeric)
    matched_brush_ids = torch.zeros_like(target_brush_ids)
    matched_mask = torch.zeros_like(target_padding_mask, dtype=torch.bool)
    for batch_index in range(batch_size):
        target_indices = torch.nonzero(~target_padding_mask[batch_index], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        pred = pred_numeric[batch_index]
        target = target_numeric[batch_index, target_indices]
        cost = (torch.abs(pred.unsqueeze(1) - target.unsqueeze(0)) * weights.view(1, 1, field_count)).sum(dim=-1)
        pair_cost, flat_indices = torch.sort(cost.reshape(-1), stable=True)
        used_pred: set[int] = set()
        used_target: set[int] = set()
        for flat_index in flat_indices.tolist():
            pred_index = flat_index // target_indices.numel()
            local_target_index = flat_index % target_indices.numel()
            if pred_index in used_pred or local_target_index in used_target:
                continue
            original_target_index = int(target_indices[local_target_index].item())
            matched_numeric[batch_index, pred_index] = target_numeric[batch_index, original_target_index]
            matched_brush_ids[batch_index, pred_index] = target_brush_ids[batch_index, original_target_index]
            matched_mask[batch_index, pred_index] = True
            used_pred.add(pred_index)
            used_target.add(local_target_index)
            if len(used_target) == target_indices.numel() or len(used_pred) == slot_count:
                break
    return matched_numeric, matched_brush_ids, matched_mask


def compute_anti_dot_loss(
    pred_numeric: torch.Tensor,
    pred_present_logits: torch.Tensor,
    patch_tensor: torch.Tensor,
    min_area_pixels: float = 8.0,
) -> torch.Tensor:
    """Penalize high-present strokes whose rendered footprint is too dot-like."""

    if pred_present_logits.shape != pred_numeric.shape[:2]:
        raise ValueError("pred_present_logits must have shape [batch, seq]")
    patch_area = float(patch_tensor.shape[-1] * patch_tensor.shape[-2])
    present = torch.sigmoid(pred_present_logits)
    area_pixels = (pred_numeric[..., 3].clamp_min(0.0) * pred_numeric[..., 4].clamp_min(0.0) * patch_area)
    smallness = ((min_area_pixels - area_pixels) / max(min_area_pixels, 1e-6)).clamp_min(0.0)
    return (present * smallness).mean()


def compute_color_clamp_loss(
    pred_numeric: torch.Tensor,
    pred_present_logits: torch.Tensor,
    patch_tensor: torch.Tensor,
    brightness_margin: float = 0.12,
) -> torch.Tensor:
    """Discourage bright invented marks where the target patch is not bright."""

    if patch_tensor.ndim != 4 or patch_tensor.shape[1] < 10:
        raise ValueError("patch_tensor must have shape [batch, 10, height, width]")
    target = patch_tensor[:, 3:6].to(pred_numeric.dtype)
    edit_mask = patch_tensor[:, 9:10].to(pred_numeric.dtype)
    denominator = (edit_mask.sum(dim=(2, 3), keepdim=False).clamp_min(1.0) * target.shape[1]).view(-1, 1)
    target_brightness = (target * edit_mask).sum(dim=(1, 2, 3), keepdim=False).view(-1, 1) / denominator
    pred_brightness = pred_numeric[..., 6:9].clamp(0.0, 1.0).mean(dim=-1)
    present = torch.sigmoid(pred_present_logits)
    overshoot = (pred_brightness - target_brightness - brightness_margin).clamp_min(0.0)
    return (present * overshoot).mean()


def compute_assigned_size_distribution_loss(
    pred_numeric: torch.Tensor,
    matched_numeric: torch.Tensor,
    matched_mask: torch.Tensor,
) -> torch.Tensor:
    """Match angle/size statistics for assigned slots to avoid repeated default marks."""

    if pred_numeric.shape != matched_numeric.shape:
        raise ValueError("pred_numeric and matched_numeric shapes must match")
    if matched_mask.shape != pred_numeric.shape[:2]:
        raise ValueError("matched_mask must have shape [batch, seq]")
    fields = torch.tensor([2, 3, 4], device=pred_numeric.device, dtype=torch.long)
    valid = matched_mask.unsqueeze(-1).to(pred_numeric.dtype)
    if int(matched_mask.sum().item()) == 0:
        return pred_numeric.sum() * 0.0
    pred = pred_numeric.index_select(-1, fields)
    target = matched_numeric.index_select(-1, fields)
    counts = valid.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred * valid).sum(dim=1) / counts
    target_mean = (target * valid).sum(dim=1) / counts
    pred_var = (((pred - pred_mean.unsqueeze(1)) * valid) ** 2).sum(dim=1) / counts
    target_var = (((target - target_mean.unsqueeze(1)) * valid) ** 2).sum(dim=1) / counts
    pred_std = torch.sqrt(pred_var.clamp_min(1e-8))
    target_std = torch.sqrt(target_var.clamp_min(1e-8))
    active = matched_mask.any(dim=1).to(pred_numeric.dtype).view(-1, 1)
    loss = F.smooth_l1_loss(pred_mean, target_mean, reduction="none") + F.smooth_l1_loss(
        pred_std,
        target_std,
        reduction="none",
    )
    return (loss * active).sum() / (active.sum().clamp_min(1.0) * len(fields))


def compute_visual_patch_loss(
    prediction: VisualDeltaPredictionOutput,
    batch: VisualDeltaBatch,
    training_renderer: str = TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    draft = batch.patch_tensor[:, 0:3].to(prediction.pred_numeric.dtype)
    target = batch.patch_tensor[:, 3:6].to(prediction.pred_numeric.dtype)
    edit_mask = batch.patch_tensor[:, 9:10].to(prediction.pred_numeric.dtype)
    predicted = render_training_strokes(
        draft,
        prediction.pred_numeric,
        prediction.pred_present_logits,
        training_renderer=training_renderer,
    )
    inside_denominator = edit_mask.sum().clamp_min(1.0) * draft.shape[1]
    outside_mask = 1.0 - edit_mask
    outside_denominator = outside_mask.sum().clamp_min(1.0) * draft.shape[1]
    image = (torch.abs(predicted - target) * edit_mask).sum() / inside_denominator
    preservation = (torch.abs(predicted - draft) * outside_mask).sum() / outside_denominator
    gradient = compute_gradient_loss(predicted, target, edit_mask)
    edge = compute_edge_alignment_loss(draft, predicted, target, edit_mask)
    low_frequency = compute_low_frequency_loss(predicted, target, edit_mask)
    return image, preservation, gradient, edge, low_frequency


def render_training_strokes(
    draft: torch.Tensor,
    numeric: torch.Tensor,
    present_logits: torch.Tensor,
    training_renderer: str = TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
) -> torch.Tensor:
    if training_renderer == TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT:
        return render_paint_transformer_soft_strokes(draft, numeric, present_logits)
    if training_renderer == TRAINING_RENDERER_SOFT_ELLIPSE:
        return render_soft_strokes(draft, numeric, present_logits)
    raise ValueError(
        f"unsupported training_renderer {training_renderer!r}; "
        f"expected one of {', '.join(SUPPORTED_TRAINING_RENDERERS)}"
    )


def compute_gradient_loss(predicted: torch.Tensor, target: torch.Tensor, edit_mask: torch.Tensor) -> torch.Tensor:
    pred_grad = sobel_magnitude(predicted)
    target_grad = sobel_magnitude(target)
    mask = _resize_mask(edit_mask, pred_grad.shape[-2:])
    denominator = mask.sum().clamp_min(1.0)
    return (torch.abs(pred_grad - target_grad) * mask).sum() / denominator


def compute_edge_alignment_loss(
    draft: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
    edge_threshold: float = 0.08,
) -> torch.Tensor:
    target_delta_edges = sobel_magnitude(torch.abs(target - draft)).detach()
    predicted_delta_edges = sobel_magnitude(torch.abs(predicted - draft))
    mask = _resize_mask(edit_mask, predicted_delta_edges.shape[-2:])
    target_edges = ((target_delta_edges > edge_threshold).to(predicted_delta_edges.dtype) * mask).detach()
    predicted_edges = (1.0 - torch.exp(-predicted_delta_edges * 2.0)).clamp(1e-4, 1.0 - 1e-4)
    positive_weight = (mask.sum() / target_edges.sum().clamp_min(1.0)).clamp(1.0, 16.0)
    edge_loss = F.binary_cross_entropy(
        predicted_edges,
        target_edges,
        reduction="none",
    )
    weights = torch.where(target_edges > 0.0, positive_weight, torch.ones_like(target_edges))
    return (edge_loss * weights * mask).sum() / (weights * mask).sum().clamp_min(1.0)


def compute_low_frequency_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
    output_size: int = 16,
) -> torch.Tensor:
    pred_low = F.adaptive_avg_pool2d(predicted, (output_size, output_size))
    target_low = F.adaptive_avg_pool2d(target, (output_size, output_size))
    mask_low = F.adaptive_avg_pool2d(edit_mask, (output_size, output_size))
    denominator = mask_low.sum().clamp_min(1.0) * predicted.shape[1]
    return (torch.abs(pred_low - target_low) * mask_low).sum() / denominator


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("image must have shape [batch, channels, height, width]")
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def _resize_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if mask.shape[-2:] == shape:
        return mask
    return F.interpolate(mask, size=shape, mode="nearest")


def compute_present_recall_loss(pred_present_logits: torch.Tensor, target_present_count: torch.Tensor) -> torch.Tensor:
    pred_count = torch.sigmoid(pred_present_logits).sum(dim=1)
    missing = (target_present_count - pred_count).clamp_min(0.0)
    return (missing / target_present_count.clamp_min(1.0)).mean()


def compute_present_count_loss(pred_present_logits: torch.Tensor, target_present_count: torch.Tensor) -> torch.Tensor:
    pred_count = torch.sigmoid(pred_present_logits).sum(dim=1)
    return (torch.abs(pred_count - target_present_count) / target_present_count.clamp_min(1.0)).mean()


def render_soft_strokes(
    draft: torch.Tensor,
    numeric: torch.Tensor,
    present_logits: torch.Tensor,
    min_sigma: float = 1.0 / 64.0,
) -> torch.Tensor:
    """Differentiably render soft ellipse strokes over a patch-sized draft tensor."""

    if draft.ndim != 4 or draft.shape[1] != 3:
        raise ValueError("draft must have shape [batch, 3, height, width]")
    if numeric.ndim != 3 or numeric.shape[-1] != 9:
        raise ValueError("numeric must have shape [batch, seq, 9]")
    if present_logits.shape != numeric.shape[:2]:
        raise ValueError("present_logits must have shape [batch, seq]")
    batch_size, _, height, width = draft.shape
    y_coords = (torch.arange(height, device=draft.device, dtype=draft.dtype) + 0.5) / height
    x_coords = (torch.arange(width, device=draft.device, dtype=draft.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)
    image = draft
    present = torch.sigmoid(present_logits).view(batch_size, -1, 1, 1)
    opacity = numeric[..., 5].clamp(0.0, 1.0).view(batch_size, -1, 1, 1)
    alpha_scale = (present * opacity).clamp(0.0, 1.0)
    centers_x = numeric[..., 0].clamp(0.0, 1.0).view(batch_size, -1, 1, 1)
    centers_y = numeric[..., 1].clamp(0.0, 1.0).view(batch_size, -1, 1, 1)
    angle = numeric[..., 2].clamp(0.0, 1.0).view(batch_size, -1, 1, 1) * torch.pi
    radius_x = (numeric[..., 3].clamp_min(min_sigma) * 0.5).view(batch_size, -1, 1, 1)
    radius_y = (numeric[..., 4].clamp_min(min_sigma) * 0.5).view(batch_size, -1, 1, 1)
    color = numeric[..., 6:9].clamp(0.0, 1.0)
    dx = xx - centers_x
    dy = yy - centers_y
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotated_x = dx * cos_angle + dy * sin_angle
    rotated_y = -dx * sin_angle + dy * cos_angle
    mask = torch.exp(-0.5 * ((rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2))
    alpha = (mask * alpha_scale).clamp(0.0, 1.0)
    for slot_index in range(numeric.shape[1]):
        slot_alpha = alpha[:, slot_index : slot_index + 1]
        slot_color = color[:, slot_index].view(batch_size, 3, 1, 1)
        image = image * (1.0 - slot_alpha) + slot_color * slot_alpha
    return image.clamp(0.0, 1.0)


def _slot_anchors(slot_count: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if slot_count == 4:
        coarse = _grid_anchors(1, 1, device, dtype)
        detail = _grid_anchors(1, 3, device, dtype)
    elif slot_count == 64:
        coarse = _grid_anchors(4, 4, device, dtype)
        detail = _grid_anchors(6, 8, device, dtype)
    else:
        coarse_grid_size = max(1, int((slot_count * 0.25) ** 0.5))
        coarse = _grid_anchors(coarse_grid_size, coarse_grid_size, device, dtype)
        rows = int(slot_count**0.5)
        while rows > 1 and slot_count % rows != 0:
            rows -= 1
        detail = _grid_anchors(rows, slot_count // rows, device, dtype)
    anchors = torch.cat([coarse, detail], dim=0)[:slot_count]
    roles = torch.cat(
        [
            torch.zeros(coarse.shape[0], device=device, dtype=torch.long),
            torch.ones(detail.shape[0], device=device, dtype=torch.long),
        ],
        dim=0,
    )[:slot_count]
    return anchors, roles


def _grid_anchors(rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y_coords = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows
    x_coords = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols
    y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def _target_roles_for_numeric(target: torch.Tensor) -> torch.Tensor:
    length = target[..., 3]
    width = target[..., 4]
    area = length * width
    coarse = (length >= 0.09) | (width >= 0.055) | (area >= 0.005)
    return torch.where(coarse, torch.zeros_like(length, dtype=torch.long), torch.ones_like(length, dtype=torch.long))


def _target_assignment_order(target: torch.Tensor) -> torch.Tensor:
    area = target[..., 3] * target[..., 4]
    # torch.argsort is stable here so equal-area targets preserve their original stroke order.
    return torch.argsort(area, descending=True, stable=True)
