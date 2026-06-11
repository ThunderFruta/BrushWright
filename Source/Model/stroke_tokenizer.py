"""Tokenization utilities for BrushWright stroke programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from Source.Renderer.stroke_schema import StrokeProgram, load_stroke_program_json


NUMERIC_FIELDS = ("x", "y", "angle", "length", "width", "opacity", "r", "g", "b")
PAD_BRUSH_ID = 0
UNK_BRUSH_ID = 1
DEFAULT_BRUSH_VOCAB = (
    "paint_transformer_rect",
    "flat_oil",
    "flat_vector",
    "mono_line",
    "neon_line",
    "soft_marker",
)


@dataclass(frozen=True)
class StrokeTokenBatch:
    numeric: torch.Tensor
    brush_ids: torch.Tensor
    padding_mask: torch.Tensor
    lengths: torch.Tensor


class StrokeTokenizer:
    """Convert validated stroke programs into padded tensor batches."""

    def __init__(
        self,
        max_strokes: int = 3072,
        brush_vocab: Sequence[str] = DEFAULT_BRUSH_VOCAB,
        allow_truncation: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        if max_strokes <= 0:
            raise ValueError("max_strokes must be positive")
        self.max_strokes = max_strokes
        self.allow_truncation = allow_truncation
        self.device = torch.device(device) if device is not None else None
        self.brush_to_id = {brush: index + 2 for index, brush in enumerate(brush_vocab)}
        self.id_to_brush = {
            PAD_BRUSH_ID: "<PAD>",
            UNK_BRUSH_ID: "<UNK>",
            **{index: brush for brush, index in self.brush_to_id.items()},
        }

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_brush) + 1

    @property
    def numeric_dim(self) -> int:
        return len(NUMERIC_FIELDS)

    def encode_program(self, program: StrokeProgram | dict[str, Any]) -> StrokeTokenBatch:
        return self.encode_programs([program])

    def encode_programs(
        self,
        programs: Sequence[StrokeProgram | dict[str, Any]],
        allow_truncation: bool | None = None,
    ) -> StrokeTokenBatch:
        if not programs:
            raise ValueError("programs must contain at least one stroke program")

        parsed_programs = [_coerce_program(program) for program in programs]
        should_truncate = self.allow_truncation if allow_truncation is None else allow_truncation
        batch_size = len(parsed_programs)
        numeric = torch.zeros(batch_size, self.max_strokes, self.numeric_dim, dtype=torch.float32, device=self.device)
        brush_ids = torch.full(
            (batch_size, self.max_strokes),
            PAD_BRUSH_ID,
            dtype=torch.long,
            device=self.device,
        )
        padding_mask = torch.ones(batch_size, self.max_strokes, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for batch_index, parsed_program in enumerate(parsed_programs):
            strokes = parsed_program.strokes
            if len(strokes) > self.max_strokes:
                if not should_truncate:
                    raise ValueError(
                        f"stroke program has {len(strokes)} strokes; max_strokes is {self.max_strokes}. "
                        "Enable truncation to encode only the leading strokes."
                    )
                strokes = strokes[: self.max_strokes]

            lengths[batch_index] = len(strokes)
            padding_mask[batch_index, : len(strokes)] = False
            for stroke_index, stroke in enumerate(strokes):
                numeric[batch_index, stroke_index] = torch.tensor(
                    (
                        stroke.x,
                        stroke.y,
                        stroke.angle,
                        stroke.length,
                        stroke.width,
                        stroke.opacity,
                        stroke.color[0],
                        stroke.color[1],
                        stroke.color[2],
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                brush_ids[batch_index, stroke_index] = self.brush_to_id.get(stroke.brush, UNK_BRUSH_ID)

        return StrokeTokenBatch(
            numeric=numeric,
            brush_ids=brush_ids,
            padding_mask=padding_mask,
            lengths=lengths,
        )


def _coerce_program(program: StrokeProgram | dict[str, Any]) -> StrokeProgram:
    if isinstance(program, StrokeProgram):
        return program
    return load_stroke_program_json(program)
