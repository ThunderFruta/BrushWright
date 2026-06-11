"""BrushWright model components."""

from Source.Model.stroke_dataset import (
    BrushWrightStrokeDataset,
    StrokeBatch,
    StrokeDatasetItem,
    collate_stroke_chunks,
    load_draft_image_tensor,
)
from Source.Model.draft_image_encoder import DraftImageEncoder, DraftImageEncoderConfig, DraftImageEncoderOutput
from Source.Model.stroke_decoder import StrokeChunkDecoder, StrokeChunkDecoderConfig, StrokePredictionOutput
from Source.Model.stroke_encoder import StrokeEncoder, StrokeEncoderConfig, StrokeEncoderOutput
from Source.Model.stroke_loss import StrokeLossOutput, compute_distribution_loss, compute_render_loss, compute_stroke_loss, match_stroke_targets
from Source.Model.stroke_predictor import BrushWrightStrokePredictor
from Source.Model.stroke_tokenizer import (
    DEFAULT_BRUSH_VOCAB,
    NUMERIC_FIELDS,
    PAD_BRUSH_ID,
    UNK_BRUSH_ID,
    StrokeTokenBatch,
    StrokeTokenizer,
)
from Source.Model.visual_delta_dataset import (
    VisualDeltaBatch,
    VisualDeltaDatasetItem,
    VisualDeltaStrokeDataset,
    collate_visual_delta_patches,
    patch_numeric_to_global_stroke,
)
from Source.Model.visual_delta_loss import (
    SUPPORTED_TRAINING_RENDERERS,
    TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT,
    TRAINING_RENDERER_SOFT_ELLIPSE,
    VisualDeltaLossOutput,
    compute_edge_alignment_loss,
    compute_gradient_loss,
    compute_low_frequency_loss,
    compute_anti_dot_loss,
    compute_assigned_size_distribution_loss,
    compute_color_clamp_loss,
    compute_present_count_loss,
    compute_present_recall_loss,
    compute_visual_delta_loss,
    compute_visual_patch_loss,
    match_visual_delta_strokes_slot_aware,
    match_visual_delta_strokes,
    render_training_strokes,
    render_soft_strokes,
    sobel_magnitude,
)
from Source.Model.paint_transformer_soft_renderer import render_paint_transformer_soft_strokes
from Source.Model.visual_delta_predictor import (
    VisualDeltaPredictionOutput,
    VisualDeltaStrokeCompiler,
    VisualDeltaStrokeCompilerConfig,
)

__all__ = [
    "DEFAULT_BRUSH_VOCAB",
    "NUMERIC_FIELDS",
    "PAD_BRUSH_ID",
    "UNK_BRUSH_ID",
    "BrushWrightStrokeDataset",
    "BrushWrightStrokePredictor",
    "DraftImageEncoder",
    "DraftImageEncoderConfig",
    "DraftImageEncoderOutput",
    "StrokeBatch",
    "StrokeChunkDecoder",
    "StrokeChunkDecoderConfig",
    "StrokeDatasetItem",
    "StrokeEncoder",
    "StrokeEncoderConfig",
    "StrokeEncoderOutput",
    "StrokeLossOutput",
    "StrokePredictionOutput",
    "StrokeTokenBatch",
    "StrokeTokenizer",
    "StrokeTrainingConfig",
    "SUPPORTED_TRAINING_RENDERERS",
    "TRAINING_RENDERER_PAINT_TRANSFORMER_SOFT",
    "TRAINING_RENDERER_SOFT_ELLIPSE",
    "VisualDeltaBatch",
    "VisualDeltaDatasetItem",
    "VisualDeltaLossOutput",
    "VisualDeltaPredictionOutput",
    "VisualDeltaStrokeCompiler",
    "VisualDeltaStrokeCompilerConfig",
    "VisualDeltaStrokeDataset",
    "VisualDeltaTrainingConfig",
    "collate_stroke_chunks",
    "collate_visual_delta_patches",
    "compute_stroke_loss",
    "match_stroke_targets",
    "compute_distribution_loss",
    "compute_render_loss",
    "compute_edge_alignment_loss",
    "compute_gradient_loss",
    "compute_low_frequency_loss",
    "compute_anti_dot_loss",
    "compute_assigned_size_distribution_loss",
    "compute_color_clamp_loss",
    "compute_present_count_loss",
    "compute_present_recall_loss",
    "compute_visual_delta_loss",
    "compute_visual_patch_loss",
    "load_draft_image_tensor",
    "patch_numeric_to_global_stroke",
    "match_visual_delta_strokes_slot_aware",
    "match_visual_delta_strokes",
    "render_soft_strokes",
    "render_training_strokes",
    "render_paint_transformer_soft_strokes",
    "sobel_magnitude",
    "train_strokes",
    "train_visual_delta_strokes",
]


def train_strokes(config):
    from Source.Model.train_strokes import train_strokes as run_training

    return run_training(config)


def __getattr__(name):
    if name == "StrokeTrainingConfig":
        from Source.Model.train_strokes import StrokeTrainingConfig

        return StrokeTrainingConfig
    if name == "VisualDeltaTrainingConfig":
        from Source.Model.train_visual_delta_strokes import VisualDeltaTrainingConfig

        return VisualDeltaTrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def train_visual_delta_strokes(config):
    from Source.Model.train_visual_delta_strokes import train_visual_delta_strokes as run_training

    return run_training(config)
