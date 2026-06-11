"""Combined stroke-only BrushWright predictor."""

from __future__ import annotations

import torch.nn as nn
import torch

from Source.Model.draft_image_encoder import DraftImageEncoder, DraftImageEncoderConfig
from Source.Model.stroke_dataset import StrokeBatch
from Source.Model.stroke_decoder import StrokeChunkDecoder, StrokeChunkDecoderConfig, StrokePredictionOutput
from Source.Model.stroke_encoder import StrokeEncoder, StrokeEncoderConfig, StrokeEncoderOutput


class BrushWrightStrokePredictor(nn.Module):
    """Predict finishing-stroke chunks from base-stroke batches."""

    def __init__(
        self,
        encoder_config: StrokeEncoderConfig | None = None,
        decoder_config: StrokeChunkDecoderConfig | None = None,
        image_encoder_config: DraftImageEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        resolved_encoder_config = encoder_config or StrokeEncoderConfig()
        resolved_decoder_config = decoder_config or StrokeChunkDecoderConfig(
            model_dim=resolved_encoder_config.model_dim,
            num_heads=resolved_encoder_config.num_heads,
            ff_dim=resolved_encoder_config.ff_dim,
            dropout=resolved_encoder_config.dropout,
            brush_vocab=resolved_encoder_config.brush_vocab,
        )
        if resolved_encoder_config.model_dim != resolved_decoder_config.model_dim:
            raise ValueError("encoder and decoder model_dim values must match")
        self.encoder = StrokeEncoder(resolved_encoder_config)
        if image_encoder_config is not None and image_encoder_config.model_dim != resolved_encoder_config.model_dim:
            raise ValueError("image encoder and stroke encoder model_dim values must match")
        self.image_encoder = DraftImageEncoder(image_encoder_config) if image_encoder_config is not None else None
        self.decoder = StrokeChunkDecoder(resolved_decoder_config)

    def forward(self, batch: StrokeBatch) -> StrokePredictionOutput:
        encoder_output = self.encoder(
            batch.base_tokens.numeric,
            batch.base_tokens.brush_ids,
            batch.base_tokens.padding_mask,
        )
        if self.image_encoder is not None:
            if batch.draft_images is None:
                raise ValueError("draft_images are required when image conditioning is enabled")
            image_input = _image_conditioning_tensor(batch, self.image_encoder.config.input_channels)
            image_output = self.image_encoder(image_input)
            encoder_output = StrokeEncoderOutput(
                features=torch.cat([image_output.features, encoder_output.features], dim=1),
                pooled=(encoder_output.pooled + image_output.pooled) * 0.5,
                padding_mask=torch.cat([image_output.padding_mask, encoder_output.padding_mask], dim=1),
            )
        return self.decoder(encoder_output, batch.chunk_starts)


def _image_conditioning_tensor(batch: StrokeBatch, input_channels: int) -> torch.Tensor:
    if batch.draft_images is None:
        raise ValueError("draft_images are required")
    if input_channels == 3:
        return batch.draft_images
    if batch.goal_images is None or batch.error_maps is None:
        raise ValueError("goal_images and error_maps are required for 9-channel conditioning")
    if input_channels != 9:
        raise ValueError("image encoder input_channels must be 3 or 9")
    return torch.cat([batch.draft_images, batch.goal_images, batch.error_maps], dim=1)
