"""Small draft-image encoder for BrushWright stroke prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DraftImageEncoderConfig:
    model_dim: int = 256
    hidden_dim: int = 64
    grid_size: int = 8
    dropout: float = 0.1
    input_channels: int = 3


@dataclass(frozen=True)
class DraftImageEncoderOutput:
    features: torch.Tensor
    pooled: torch.Tensor
    padding_mask: torch.Tensor


class DraftImageEncoder(nn.Module):
    """Encode a draft RGB image into cross-attention tokens."""

    def __init__(self, config: DraftImageEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DraftImageEncoderConfig()
        if self.config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.input_channels <= 0:
            raise ValueError("input_channels must be positive")

        hidden = self.config.hidden_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(self.config.input_channels, hidden, kernel_size=7, stride=2, padding=3),
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
        self.dropout = nn.Dropout(self.config.dropout)
        self.output_norm = nn.LayerNorm(self.config.model_dim)

    def forward(self, draft_images: torch.Tensor) -> DraftImageEncoderOutput:
        if draft_images.ndim != 4 or draft_images.shape[1] != self.config.input_channels:
            raise ValueError(
                "draft_images must have shape "
                f"[batch, {self.config.input_channels}, height, width]"
            )
        hidden = self.backbone(draft_images.float())
        hidden = hidden.flatten(2).transpose(1, 2)
        features = self.projection(hidden) + self.position_embedding
        features = self.output_norm(self.dropout(features))
        padding_mask = torch.zeros(
            features.shape[:2],
            dtype=torch.bool,
            device=features.device,
        )
        pooled = features.mean(dim=1)
        return DraftImageEncoderOutput(features=features, pooled=pooled, padding_mask=padding_mask)
