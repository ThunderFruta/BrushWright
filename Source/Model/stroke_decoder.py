"""Stroke chunk decoder for BrushWright V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Source.Model.stroke_encoder import StrokeEncoderOutput
from Source.Model.stroke_tokenizer import DEFAULT_BRUSH_VOCAB, NUMERIC_FIELDS


@dataclass(frozen=True)
class StrokeChunkDecoderConfig:
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    chunk_size: int = 64
    max_chunks: int = 24
    brush_vocab: tuple[str, ...] = DEFAULT_BRUSH_VOCAB
    query_mode: str = "learned"
    spatial_grid_size: int = 8
    xy_offset_scale: float = 0.75
    min_length: float = 0.006
    max_length: float = 0.03
    min_width: float = 0.005
    max_width: float = 0.035


@dataclass(frozen=True)
class StrokePredictionOutput:
    pred_numeric: torch.Tensor
    pred_brush_logits: torch.Tensor


class StrokeChunkDecoder(nn.Module):
    """Predict one finishing-stroke chunk from encoded base strokes."""

    def __init__(self, config: StrokeChunkDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or StrokeChunkDecoderConfig()
        if self.config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.config.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if self.config.model_dim % self.config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.config.query_mode not in ("learned", "spatial"):
            raise ValueError("query_mode must be 'learned' or 'spatial'")
        if self.config.spatial_grid_size <= 0:
            raise ValueError("spatial_grid_size must be positive")
        if not 0.0 < self.config.min_length < self.config.max_length <= 1.0:
            raise ValueError("length bounds must satisfy 0 < min_length < max_length <= 1")
        if not 0.0 < self.config.min_width < self.config.max_width <= 1.0:
            raise ValueError("width bounds must satisfy 0 < min_width < max_width <= 1")

        self.query_embedding = nn.Embedding(self.config.chunk_size, self.config.model_dim)
        self.chunk_embedding = nn.Embedding(self.config.max_chunks, self.config.model_dim)
        self.spatial_projection = nn.Linear(self.config.model_dim, self.config.model_dim)
        self.anchor_embedding = nn.Linear(2, self.config.model_dim)
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
        self._initialize_heads()

    def forward(self, encoder_output: StrokeEncoderOutput, chunk_starts: torch.Tensor) -> StrokePredictionOutput:
        if encoder_output.features.ndim != 3:
            raise ValueError("encoder_output.features must have shape [batch, seq, model_dim]")
        if encoder_output.features.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"encoder feature dim {encoder_output.features.shape[-1]} does not match decoder model_dim "
                f"{self.config.model_dim}"
            )
        if chunk_starts.ndim != 1 or chunk_starts.shape[0] != encoder_output.features.shape[0]:
            raise ValueError("chunk_starts must have shape [batch]")

        batch_size = encoder_output.features.shape[0]
        chunk_ids = torch.div(chunk_starts.to(encoder_output.features.device), self.config.chunk_size, rounding_mode="floor")
        chunk_ids = chunk_ids.clamp(min=0, max=self.config.max_chunks - 1)
        anchors = None
        if self.config.query_mode == "spatial":
            queries, anchors = self._spatial_queries(encoder_output.features, chunk_ids)
        else:
            query_ids = torch.arange(self.config.chunk_size, device=encoder_output.features.device)
            queries = self.query_embedding(query_ids).unsqueeze(0).expand(batch_size, self.config.chunk_size, -1)
            queries = queries + self.chunk_embedding(chunk_ids).unsqueeze(1)
        decoded = self.decoder(
            tgt=queries,
            memory=encoder_output.features,
            memory_key_padding_mask=encoder_output.padding_mask,
        )
        decoded = self.output_norm(decoded)
        numeric = self._decode_numeric(decoded, anchors)
        return StrokePredictionOutput(
            pred_numeric=numeric,
            pred_brush_logits=self.brush_head(decoded),
        )

    def _spatial_queries(self, features: torch.Tensor, chunk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid_tokens = self.config.spatial_grid_size * self.config.spatial_grid_size
        if features.shape[1] < grid_tokens:
            raise ValueError("spatial query mode requires image features to be prepended to encoder output")
        image_features = features[:, :grid_tokens]
        slot_ids = torch.arange(self.config.chunk_size, device=features.device)
        grid_indices = (slot_ids + chunk_ids.unsqueeze(1) * self.config.chunk_size) % grid_tokens
        gathered = image_features.gather(
            1,
            grid_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
        )
        anchors = _grid_anchors(
            grid_size=self.config.spatial_grid_size,
            device=features.device,
            dtype=features.dtype,
        )
        gathered_anchors = anchors[grid_indices]
        queries = (
            self.spatial_projection(gathered)
            + self.query_embedding(slot_ids).unsqueeze(0)
            + self.chunk_embedding(chunk_ids).unsqueeze(1)
            + self.anchor_embedding(gathered_anchors)
        )
        return queries, gathered_anchors

    def _decode_numeric(self, decoded: torch.Tensor, anchors: torch.Tensor | None) -> torch.Tensor:
        raw = self.numeric_head(decoded)
        values = torch.sigmoid(raw)
        if anchors is None:
            length = _scale_sigmoid(raw[..., 3:4], self.config.min_length, self.config.max_length)
            width = _scale_sigmoid(raw[..., 4:5], self.config.min_width, self.config.max_width)
            return torch.cat(
                [
                    values[..., 0:3],
                    length,
                    width,
                    values[..., 5:],
                ],
                dim=-1,
            )

        xy_offsets = torch.tanh(raw[..., 0:2]) * (self.config.xy_offset_scale / self.config.spatial_grid_size)
        xy = (anchors + xy_offsets).clamp(0.0, 1.0)
        length = _scale_sigmoid(raw[..., 3:4], self.config.min_length, self.config.max_length)
        width = _scale_sigmoid(raw[..., 4:5], self.config.min_width, self.config.max_width)
        return torch.cat(
            [
                xy,
                values[..., 2:3],
                length,
                width,
                values[..., 5:],
            ],
            dim=-1,
        )

    def _initialize_heads(self) -> None:
        nn.init.normal_(self.numeric_head.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.numeric_head.bias.copy_(
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        _logit(0.4),
                        _inverse_scaled_sigmoid(0.016, self.config.min_length, self.config.max_length),
                        _inverse_scaled_sigmoid(0.012, self.config.min_width, self.config.max_width),
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


def _scale_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + torch.sigmoid(raw) * (high - low)


def _logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return torch.logit(torch.tensor(clipped)).item()


def _inverse_scaled_sigmoid(value: float, low: float, high: float) -> float:
    normalized = (value - low) / (high - low)
    return _logit(normalized)
