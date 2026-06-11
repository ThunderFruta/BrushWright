"""Transformer encoder for BrushWright stroke token batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Source.Model.stroke_tokenizer import DEFAULT_BRUSH_VOCAB, NUMERIC_FIELDS, PAD_BRUSH_ID


@dataclass(frozen=True)
class StrokeEncoderConfig:
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_strokes: int = 3072
    brush_vocab: tuple[str, ...] = DEFAULT_BRUSH_VOCAB


@dataclass(frozen=True)
class StrokeEncoderOutput:
    features: torch.Tensor
    pooled: torch.Tensor
    padding_mask: torch.Tensor


class StrokeEncoder(nn.Module):
    """Encode padded stroke tokens into contextual stroke features."""

    def __init__(self, config: StrokeEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or StrokeEncoderConfig()
        if self.config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.config.max_strokes <= 0:
            raise ValueError("max_strokes must be positive")
        if self.config.model_dim % self.config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        vocab_size = len(self.config.brush_vocab) + 2
        self.numeric_projection = nn.Linear(len(NUMERIC_FIELDS), self.config.model_dim)
        self.brush_embedding = nn.Embedding(vocab_size, self.config.model_dim, padding_idx=PAD_BRUSH_ID)
        self.position_embedding = nn.Embedding(self.config.max_strokes, self.config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.config.model_dim)

    def forward(
        self,
        numeric: torch.Tensor,
        brush_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> StrokeEncoderOutput:
        if numeric.ndim != 3 or numeric.shape[-1] != len(NUMERIC_FIELDS):
            raise ValueError(f"numeric must have shape [batch, seq, {len(NUMERIC_FIELDS)}]")
        if brush_ids.shape != numeric.shape[:2]:
            raise ValueError("brush_ids must have shape [batch, seq]")
        if padding_mask.shape != numeric.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, seq]")
        if numeric.shape[1] > self.config.max_strokes:
            raise ValueError(f"sequence length {numeric.shape[1]} exceeds max_strokes {self.config.max_strokes}")

        batch_size, sequence_length, _ = numeric.shape
        positions = torch.arange(sequence_length, device=numeric.device).unsqueeze(0).expand(batch_size, sequence_length)
        hidden = (
            self.numeric_projection(numeric)
            + self.brush_embedding(brush_ids)
            + self.position_embedding(positions)
        )
        features = self.encoder(hidden, src_key_padding_mask=padding_mask)
        features = self.output_norm(features)
        pooled = _masked_mean(features, padding_mask)
        return StrokeEncoderOutput(features=features, pooled=pooled, padding_mask=padding_mask)


def _masked_mean(features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = ~padding_mask
    weights = valid_mask.unsqueeze(-1).to(features.dtype)
    summed = (features * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts
