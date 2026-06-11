"""Training-only PaintTransformer-aligned soft stroke renderer.

This module owns differentiable patch rendering for visual-delta training. It
uses the Paint Transformer brush masks and affine placement math, but keeps the
binary alpha threshold and morphology steps soft so gradients can reach the
stroke heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from Source.PaintTransformerReference.rendering import load_meta_brushes


MIN_RENDER_SIZE = 0.001


def render_paint_transformer_soft_strokes(
    draft: torch.Tensor,
    numeric: torch.Tensor,
    present_logits: torch.Tensor,
) -> torch.Tensor:
    """Render BrushWright patch-local strokes with PaintTransformer-style masks."""

    if draft.ndim != 4 or draft.shape[1] != 3:
        raise ValueError("draft must have shape [batch, 3, height, width]")
    if numeric.ndim != 3 or numeric.shape[-1] != 9:
        raise ValueError("numeric must have shape [batch, seq, 9]")
    if present_logits.shape != numeric.shape[:2]:
        raise ValueError("present_logits must have shape [batch, seq]")

    batch_size, _, height, width = draft.shape
    slot_count = numeric.shape[1]
    if slot_count == 0:
        return draft.clamp(0.0, 1.0)

    param = _numeric_to_paint_transformer_param(numeric)
    foreground, alpha = _param_to_soft_stroke(
        param.reshape(batch_size * slot_count, 8),
        height=height,
        width=width,
    )
    foreground = foreground.view(batch_size, slot_count, 3, height, width)
    alpha = alpha.view(batch_size, slot_count, 1, height, width)
    present = torch.sigmoid(present_logits).view(batch_size, slot_count, 1, 1, 1)
    opacity = numeric[..., 5].clamp(0.0, 1.0).view(batch_size, slot_count, 1, 1, 1)
    alpha = (alpha * present * opacity).clamp(0.0, 1.0)

    image = draft
    for slot_index in range(slot_count):
        slot_alpha = alpha[:, slot_index]
        slot_foreground = foreground[:, slot_index]
        image = slot_foreground * slot_alpha + image * (1.0 - slot_alpha)
    return image.clamp(0.0, 1.0)


def _numeric_to_paint_transformer_param(numeric: torch.Tensor) -> torch.Tensor:
    x0 = numeric[..., 0].clamp(0.0, 1.0)
    y0 = numeric[..., 1].clamp(0.0, 1.0)
    theta = (numeric[..., 2].clamp(0.0, 1.0) * 2.0) % 1.0
    stroke_width = numeric[..., 3].clamp_min(MIN_RENDER_SIZE)
    stroke_height = numeric[..., 4].clamp_min(MIN_RENDER_SIZE)
    color = numeric[..., 6:9].clamp(0.0, 1.0)
    return torch.cat(
        [
            x0.unsqueeze(-1),
            y0.unsqueeze(-1),
            stroke_width.unsqueeze(-1),
            stroke_height.unsqueeze(-1),
            theta.unsqueeze(-1),
            color,
        ],
        dim=-1,
    )


def _param_to_soft_stroke(param: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if param.ndim != 2 or param.shape[-1] != 8:
        raise ValueError("param must have shape [batch, 8]")

    meta_brushes = load_meta_brushes(param.device).to(dtype=param.dtype)
    meta_brushes = F.interpolate(meta_brushes, (height, width), mode="bilinear", align_corners=False)
    x0, y0, stroke_width, stroke_height, theta = [item.squeeze(-1) for item in torch.split(param[:, :5], 1, dim=1)]
    red, green, blue = [item.squeeze(-1) for item in torch.split(param[:, 5:], 1, dim=1)]
    stroke_width = stroke_width.clamp_min(MIN_RENDER_SIZE)
    stroke_height = stroke_height.clamp_min(MIN_RENDER_SIZE)
    pi = torch.acos(torch.tensor(-1.0, device=param.device, dtype=param.dtype))
    sin_theta = torch.sin(pi * theta)
    cos_theta = torch.cos(pi * theta)

    brush_indices = torch.where(
        stroke_height > stroke_width,
        torch.zeros_like(stroke_width, dtype=torch.long),
        torch.ones_like(stroke_width, dtype=torch.long),
    )
    brush = meta_brushes[brush_indices]

    warp_00 = cos_theta / stroke_width
    warp_01 = sin_theta * height / (width * stroke_width)
    warp_02 = (1 - 2 * x0) * cos_theta / stroke_width + (1 - 2 * y0) * sin_theta * height / (
        width * stroke_width
    )
    warp_10 = -sin_theta * width / (height * stroke_height)
    warp_11 = cos_theta / stroke_height
    warp_12 = (1 - 2 * y0) * cos_theta / stroke_height - (1 - 2 * x0) * sin_theta * width / (
        height * stroke_height
    )
    warp = torch.stack(
        [
            torch.stack([warp_00, warp_01, warp_02], dim=1),
            torch.stack([warp_10, warp_11, warp_12], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(warp, [param.shape[0], 3, height, width], align_corners=False)
    brush = F.grid_sample(brush, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0.0, 1.0)
    alpha = brush
    color_map = torch.stack([red, green, blue], dim=1).view(param.shape[0], 3, 1, 1)
    foreground = brush.repeat(1, 3, 1, 1) * color_map
    return foreground, alpha
