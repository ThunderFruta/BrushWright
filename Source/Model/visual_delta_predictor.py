"""Visual-delta-to-stroke compiler model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Source.Model.stroke_tokenizer import DEFAULT_BRUSH_VOCAB, NUMERIC_FIELDS
from Source.Model.visual_delta_dataset import VisualDeltaBatch


DEFAULT_COARSE_MIN_LENGTH = 0.014
DEFAULT_COARSE_MAX_LENGTH = 0.12
DEFAULT_COARSE_MIN_WIDTH = 0.008
DEFAULT_COARSE_MAX_WIDTH = 0.07
DEFAULT_DETAIL_MIN_LENGTH = 0.012
DEFAULT_DETAIL_MAX_LENGTH = 0.18
DEFAULT_DETAIL_MIN_WIDTH = 0.008
DEFAULT_DETAIL_MAX_WIDTH = 0.16
DEFAULT_MODEL_DIM = 768
DEFAULT_HIDDEN_DIM = 192
DEFAULT_NUM_LAYERS = 10
DEFAULT_NUM_HEADS = 12
DEFAULT_FF_DIM = 3072
DEFAULT_GRID_SIZE = 16
DEFAULT_MAX_STROKES = 512
DEFAULT_COARSE_GRID_SIZE = 11
DEFAULT_DETAIL_GRID_ROWS = 16
DEFAULT_DETAIL_GRID_COLS = 32


@dataclass(frozen=True)
class VisualDeltaStrokeCompilerConfig:
    model_dim: int = DEFAULT_MODEL_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    ff_dim: int = DEFAULT_FF_DIM
    dropout: float = 0.1
    patch_channels: int = 10
    grid_size: int = DEFAULT_GRID_SIZE
    max_strokes: int = DEFAULT_MAX_STROKES
    coarse_grid_size: int = DEFAULT_COARSE_GRID_SIZE
    detail_grid_rows: int = DEFAULT_DETAIL_GRID_ROWS
    detail_grid_cols: int = DEFAULT_DETAIL_GRID_COLS
    coarse_min_length: float = DEFAULT_COARSE_MIN_LENGTH
    coarse_max_length: float = DEFAULT_COARSE_MAX_LENGTH
    coarse_min_width: float = DEFAULT_COARSE_MIN_WIDTH
    coarse_max_width: float = DEFAULT_COARSE_MAX_WIDTH
    detail_min_length: float = DEFAULT_DETAIL_MIN_LENGTH
    detail_max_length: float = DEFAULT_DETAIL_MAX_LENGTH
    detail_min_width: float = DEFAULT_DETAIL_MIN_WIDTH
    detail_max_width: float = DEFAULT_DETAIL_MAX_WIDTH
    brush_vocab: tuple[str, ...] = DEFAULT_BRUSH_VOCAB
    xy_offset_scale: float = 0.75
    min_length: float = 0.006
    max_length: float = 0.18
    min_width: float = 0.005
    max_width: float = 0.18


@dataclass(frozen=True)
class VisualDeltaPredictionOutput:
    pred_numeric: torch.Tensor
    pred_brush_logits: torch.Tensor
    pred_present_logits: torch.Tensor


class VisualDeltaStrokeCompiler(nn.Module):
    """Predict patch-local strokes from a draft/target/error/mask patch tensor."""

    def __init__(self, config: VisualDeltaStrokeCompilerConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisualDeltaStrokeCompilerConfig()
        if self.config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.model_dim % self.config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.config.patch_channels <= 0:
            raise ValueError("patch_channels must be positive")
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.max_strokes <= 0:
            raise ValueError("max_strokes must be positive")
        if self.config.coarse_grid_size < 0:
            raise ValueError("coarse_grid_size must be non-negative")
        if self.config.detail_grid_rows <= 0 or self.config.detail_grid_cols <= 0:
            raise ValueError("detail grid dimensions must be positive")
        coarse_slots = self.config.coarse_grid_size * self.config.coarse_grid_size
        detail_slots = self.config.detail_grid_rows * self.config.detail_grid_cols
        if self.config.max_strokes > coarse_slots + detail_slots:
            raise ValueError("max_strokes cannot exceed coarse_grid_size^2 + detail_grid_rows * detail_grid_cols")

        hidden = self.config.hidden_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(self.config.patch_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.config.grid_size, self.config.grid_size)),
        )
        self.projection = nn.Linear(hidden * 2, self.config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.grid_size * self.config.grid_size, self.config.model_dim)
        )

        self.query_embedding = nn.Embedding(self.config.max_strokes, self.config.model_dim)
        self.anchor_embedding = nn.Linear(2, self.config.model_dim)
        self.role_embedding = nn.Embedding(2, self.config.model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(self.config.model_dim)
        self.numeric_head = nn.Linear(self.config.model_dim, len(NUMERIC_FIELDS))
        self.brush_head = nn.Linear(self.config.model_dim, len(self.config.brush_vocab) + 2)
        self.present_head = nn.Linear(self.config.model_dim, 1)
        self._initialize_heads()

    def forward(self, batch: VisualDeltaBatch) -> VisualDeltaPredictionOutput:
        patch_tensor = batch.patch_tensor
        if patch_tensor.ndim != 4 or patch_tensor.shape[1] != self.config.patch_channels:
            raise ValueError(
                "patch_tensor must have shape "
                f"[batch, {self.config.patch_channels}, height, width]"
            )
        batch_size = patch_tensor.shape[0]
        hidden = self.backbone(patch_tensor.float())
        hidden = hidden.flatten(2).transpose(1, 2)
        memory = self.projection(hidden) + self.position_embedding
        anchors, roles = _coarse_detail_anchors(
            coarse_grid_size=self.config.coarse_grid_size,
            detail_grid_rows=self.config.detail_grid_rows,
            detail_grid_cols=self.config.detail_grid_cols,
            device=patch_tensor.device,
            dtype=patch_tensor.dtype,
        )
        anchors = anchors[: self.config.max_strokes]
        roles = roles[: self.config.max_strokes]
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        slot_ids = torch.arange(self.config.max_strokes, device=patch_tensor.device)
        memory_queries = (
            memory[:, : self.config.max_strokes, :]
            if memory.shape[1] >= self.config.max_strokes
            else memory.mean(dim=1, keepdim=True).expand(batch_size, self.config.max_strokes, -1)
        )
        queries = (
            memory_queries
            + self.query_embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
            + self.anchor_embedding(anchors)
            + self.role_embedding(roles).unsqueeze(0).expand(batch_size, -1, -1)
        )
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = self.output_norm(decoded)
        raw = self.numeric_head(decoded)
        return VisualDeltaPredictionOutput(
            pred_numeric=self._decode_numeric(raw, anchors, roles),
            pred_brush_logits=self.brush_head(decoded),
            pred_present_logits=self.present_head(decoded).squeeze(-1),
        )

    def _decode_numeric(self, raw: torch.Tensor, anchors: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        values = torch.sigmoid(raw)
        role_scale = torch.where(
            roles.to(raw.device).view(1, -1, 1) == 0,
            torch.full(
                (),
                self.config.xy_offset_scale / max(1, self.config.coarse_grid_size),
                device=raw.device,
                dtype=raw.dtype,
            ),
            torch.full((), self.config.xy_offset_scale / max(self.config.detail_grid_rows, self.config.detail_grid_cols), device=raw.device, dtype=raw.dtype),
        )
        xy_offsets = torch.tanh(raw[..., 0:2]) * role_scale
        xy = (anchors + xy_offsets).clamp(0.0, 1.0)
        coarse_mask = (roles.to(raw.device).view(1, -1, 1) == 0).to(raw.dtype)
        detail_mask = 1.0 - coarse_mask
        length = (
            _scale_sigmoid(raw[..., 3:4], self.config.coarse_min_length, self.config.coarse_max_length) * coarse_mask
            + _scale_sigmoid(raw[..., 3:4], self.config.detail_min_length, self.config.detail_max_length) * detail_mask
        )
        width = (
            _scale_sigmoid(raw[..., 4:5], self.config.coarse_min_width, self.config.coarse_max_width) * coarse_mask
            + _scale_sigmoid(raw[..., 4:5], self.config.detail_min_width, self.config.detail_max_width) * detail_mask
        )
        return torch.cat([xy, values[..., 2:3], length, width, values[..., 5:]], dim=-1)

    def _initialize_heads(self) -> None:
        nn.init.zeros_(self.numeric_head.weight)
        nn.init.zeros_(self.present_head.weight)
        nn.init.constant_(self.present_head.bias, -1.0)
        with torch.no_grad():
            self.numeric_head.bias.copy_(
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        _logit(0.4),
                        0.0,
                        0.0,
                        _logit(0.98),
                        _logit(0.45),
                        _logit(0.45),
                        _logit(0.38),
                    ],
                    dtype=self.numeric_head.bias.dtype,
                )
            )


def _grid_anchors(grid_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def _rect_grid_anchors(rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y_coords = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows
    x_coords = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols
    y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def _coarse_detail_anchors(
    coarse_grid_size: int,
    detail_grid_rows: int,
    detail_grid_cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse = (
        _grid_anchors(coarse_grid_size, device=device, dtype=dtype)
        if coarse_grid_size > 0
        else torch.zeros(0, 2, device=device, dtype=dtype)
    )
    detail = _rect_grid_anchors(detail_grid_rows, detail_grid_cols, device=device, dtype=dtype)
    anchors = torch.cat([coarse, detail], dim=0)
    roles = torch.cat(
        [
            torch.zeros(coarse.shape[0], device=device, dtype=torch.long),
            torch.ones(detail.shape[0], device=device, dtype=torch.long),
        ],
        dim=0,
    )
    return anchors[: coarse_grid_size * coarse_grid_size + detail_grid_rows * detail_grid_cols], roles[
        : coarse_grid_size * coarse_grid_size + detail_grid_rows * detail_grid_cols
    ]


def _scale_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + torch.sigmoid(raw) * (high - low)


def _logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return torch.logit(torch.tensor(clipped)).item()


def _inverse_scaled_sigmoid(value: float, low: float, high: float) -> float:
    normalized = (value - low) / (high - low)
    return _logit(normalized)
