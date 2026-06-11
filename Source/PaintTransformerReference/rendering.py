"""Paint Transformer canvas update helpers for inference.

Adapted from https://github.com/Huage001/PaintTransformer under Apache-2.0.
The helpers keep the Paint Transformer recurrent canvas behavior available
while BrushWright still exports its own stroke schema and renders final
artifacts through Source/Renderer/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


BRUSH_DIR = Path(__file__).resolve().parent / "brushes"


def read_img(img_path: Path, img_type: str = "RGB", h: int | None = None, w: int | None = None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.Resampling.NEAREST)
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array.transpose((2, 0, 1))
    return torch.from_numpy(img_array).unsqueeze(0).float() / 255.0


def load_meta_brushes(device):
    vertical = read_img(BRUSH_DIR / "brush_large_vertical.png", "L").to(device)
    horizontal = read_img(BRUSH_DIR / "brush_large_horizontal.png", "L").to(device)
    return torch.cat([vertical, horizontal], dim=0)


def pad(img, height: int, width: int):
    _, channels, source_height, source_width = img.shape
    pad_h = (height - source_height) // 2
    pad_w = (width - source_width) // 2
    remainder_h = (height - source_height) % 2
    remainder_w = (width - source_width) % 2
    img = torch.cat(
        [
            torch.zeros((1, channels, pad_h, source_width), device=img.device),
            img,
            torch.zeros((1, channels, pad_h + remainder_h, source_width), device=img.device),
        ],
        dim=-2,
    )
    return torch.cat(
        [
            torch.zeros((1, channels, height, pad_w), device=img.device),
            img,
            torch.zeros((1, channels, height, pad_w + remainder_w), device=img.device),
        ],
        dim=-1,
    )


def crop(img, height: int, width: int):
    source_height, source_width = img.shape[-2:]
    pad_h = (source_height - height) // 2
    pad_w = (source_width - width) // 2
    remainder_h = (source_height - height) % 2
    remainder_w = (source_width - width) % 2
    return img[:, :, pad_h:source_height - pad_h - remainder_h, pad_w:source_width - pad_w - remainder_w]


def erosion(x, m: int = 1):
    batch_size, channels, height, width = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode="constant", value=1e9)
    channel = F.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(batch_size, channels, -1, height, width)
    return torch.min(channel, dim=2)[0]


def dilation(x, m: int = 1):
    batch_size, channels, height, width = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode="constant", value=-1e9)
    channel = F.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(batch_size, channels, -1, height, width)
    return torch.max(channel, dim=2)[0]


def param2stroke(param, height: int, width: int, meta_brushes):
    meta_brushes_resize = F.interpolate(meta_brushes, (height, width))
    batch_size = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, stroke_width, stroke_height, theta = [item.squeeze(-1) for item in param_list[:5]]
    red, green, blue = param_list[5:]
    pi = torch.acos(torch.tensor(-1.0, device=param.device))
    sin_theta = torch.sin(pi * theta)
    cos_theta = torch.cos(pi * theta)
    index = torch.full((batch_size,), -1, device=param.device, dtype=torch.long)
    index[stroke_height > stroke_width] = 0
    index[stroke_height <= stroke_width] = 1
    brush = meta_brushes_resize[index.long()]

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
    grid = F.affine_grid(warp, [batch_size, 3, height, width], align_corners=False)
    brush = F.grid_sample(brush, grid, align_corners=False)
    alphas = (brush > 0).float()
    brush = brush.repeat(1, 3, 1, 1)
    alphas = alphas.repeat(1, 3, 1, 1)
    color_map = torch.cat([red, green, blue], dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)
    foreground = dilation(brush * color_map)
    return foreground, erosion(alphas)


def param2img_parallel(param, decision, meta_brushes, cur_canvas):
    batch_size, patch_h, patch_w, stroke_count, _ = param.shape
    flat_param = param.view(-1, 8).contiguous()
    flat_decision = decision.view(-1).contiguous().bool()
    height, width = cur_canvas.shape[-2:]
    is_odd_y = patch_h % 2 == 1
    is_odd_x = patch_w % 2 == 1
    patch_size_y = 2 * height // patch_h
    patch_size_x = 2 * width // patch_w
    even_idx_y = torch.arange(0, patch_h, 2, device=cur_canvas.device)
    even_idx_x = torch.arange(0, patch_w, 2, device=cur_canvas.device)
    odd_idx_y = torch.arange(1, patch_h, 2, device=cur_canvas.device)
    odd_idx_x = torch.arange(1, patch_w, 2, device=cur_canvas.device)
    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid(even_idx_y, even_idx_x, indexing="ij")
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid(odd_idx_y, odd_idx_x, indexing="ij")
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid(even_idx_y, odd_idx_x, indexing="ij")
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid(odd_idx_y, even_idx_x, indexing="ij")
    cur_canvas = F.pad(
        cur_canvas,
        [patch_size_x // 4, patch_size_x // 4, patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0],
    )
    foregrounds = torch.zeros(flat_param.shape[0], 3, patch_size_y, patch_size_x, device=cur_canvas.device)
    alphas = torch.zeros(flat_param.shape[0], 3, patch_size_y, patch_size_x, device=cur_canvas.device)
    if flat_param[flat_decision, :].shape[0] > 0:
        valid_foregrounds, valid_alphas = param2stroke(
            flat_param[flat_decision, :], patch_size_y, patch_size_x, meta_brushes
        )
        foregrounds[flat_decision, :, :, :] = valid_foregrounds
        alphas[flat_decision, :, :, :] = valid_alphas
    foregrounds = foregrounds.view(-1, patch_h, patch_w, stroke_count, 3, patch_size_y, patch_size_x).contiguous()
    alphas = alphas.view(-1, patch_h, patch_w, stroke_count, 3, patch_size_y, patch_size_x).contiguous()
    decisions = flat_decision.view(-1, patch_h, patch_w, stroke_count, 1, 1, 1).contiguous()

    def partial_render(this_canvas, patch_coord_y, patch_coord_x):
        canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x), stride=(patch_size_y // 2, patch_size_x // 2))
        canvas_patch = canvas_patch.view(batch_size, 3, patch_size_y, patch_size_x, patch_h, patch_w).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
        selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
        selected_foregrounds = foregrounds[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_alphas = alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_decisions = decisions[:, patch_coord_y, patch_coord_x, :, :, :, :]
        for stroke_index in range(stroke_count):
            foreground = selected_foregrounds[:, :, :, stroke_index, :, :, :]
            alpha = selected_alphas[:, :, :, stroke_index, :, :, :]
            stroke_decision = selected_decisions[:, :, :, stroke_index, :, :, :]
            selected_canvas_patch = foreground * alpha * stroke_decision + selected_canvas_patch * (
                1 - alpha * stroke_decision
            )
        this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        return this_canvas.view(batch_size, 3, this_canvas.shape[2] * patch_size_y, this_canvas.shape[4] * patch_size_x)

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    return cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, patch_size_x // 4:-patch_size_x // 4]

