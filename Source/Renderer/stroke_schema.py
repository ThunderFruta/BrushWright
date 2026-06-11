"""Stroke program schema and validation for BrushWright V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_CANVAS_SIZE = 512
STROKE_PROGRAM_VERSION = 1
REQUIRED_STROKE_FIELDS = ("x", "y", "angle", "length", "width", "color", "opacity", "brush")


class StrokeSchemaError(ValueError):
    """Raised when a stroke program does not match the V1 schema."""


@dataclass(frozen=True)
class CanvasSpec:
    width: int = DEFAULT_CANVAS_SIZE
    height: int = DEFAULT_CANVAS_SIZE

    @classmethod
    def from_json(cls, data: Any) -> "CanvasSpec":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise StrokeSchemaError("canvas must be an object")

        width = data.get("width", DEFAULT_CANVAS_SIZE)
        height = data.get("height", DEFAULT_CANVAS_SIZE)
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise StrokeSchemaError("canvas.width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise StrokeSchemaError("canvas.height must be a positive integer")
        return cls(width=width, height=height)

    def to_json(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True)
class Stroke:
    x: float
    y: float
    angle: float
    length: float
    width: float
    color: tuple[float, float, float]
    opacity: float
    brush: str

    @classmethod
    def from_json(cls, data: Any, index: int) -> "Stroke":
        if not isinstance(data, dict):
            raise StrokeSchemaError(f"strokes[{index}] must be an object")

        for field_name in REQUIRED_STROKE_FIELDS:
            if field_name not in data:
                raise StrokeSchemaError(f"strokes[{index}].{field_name} is required")

        return cls(
            x=_normalized_float(data["x"], f"strokes[{index}].x"),
            y=_normalized_float(data["y"], f"strokes[{index}].y"),
            angle=_float_in_range(data["angle"], f"strokes[{index}].angle", 0.0, 1.0),
            length=_normalized_float(data["length"], f"strokes[{index}].length"),
            width=_normalized_float(data["width"], f"strokes[{index}].width"),
            color=_color(data["color"], f"strokes[{index}].color"),
            opacity=_normalized_float(data["opacity"], f"strokes[{index}].opacity"),
            brush=_brush(data["brush"], f"strokes[{index}].brush"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "length": self.length,
            "width": self.width,
            "color": list(self.color),
            "opacity": self.opacity,
            "brush": self.brush,
        }


@dataclass(frozen=True)
class StrokeProgram:
    version: int
    canvas: CanvasSpec
    strokes: tuple[Stroke, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Any) -> "StrokeProgram":
        if not isinstance(data, dict):
            raise StrokeSchemaError("stroke program must be an object")

        version = data.get("version")
        if version != STROKE_PROGRAM_VERSION:
            raise StrokeSchemaError(f"version must be {STROKE_PROGRAM_VERSION}")

        canvas = CanvasSpec.from_json(data.get("canvas"))
        raw_strokes = data.get("strokes")
        if not isinstance(raw_strokes, list):
            raise StrokeSchemaError("strokes must be an array")
        if not raw_strokes:
            raise StrokeSchemaError("strokes must contain at least one stroke")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise StrokeSchemaError("metadata must be an object")

        strokes = tuple(Stroke.from_json(stroke_data, index) for index, stroke_data in enumerate(raw_strokes))
        return cls(version=version, canvas=canvas, strokes=strokes, metadata=metadata)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canvas": self.canvas.to_json(),
            "strokes": [stroke.to_json() for stroke in self.strokes],
            "metadata": self.metadata,
        }


def load_stroke_program_json(data: Any) -> StrokeProgram:
    return StrokeProgram.from_json(data)


def _number(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StrokeSchemaError(f"{field_name} must be a number")
    return float(value)


def _float_in_range(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    number = _number(value, field_name)
    if number < minimum or number > maximum:
        raise StrokeSchemaError(f"{field_name} must be between {minimum} and {maximum}")
    return number


def _normalized_float(value: Any, field_name: str) -> float:
    return _float_in_range(value, field_name, 0.0, 1.0)


def _color(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise StrokeSchemaError(f"{field_name} must be an RGB array with three values")
    return tuple(_normalized_float(channel, f"{field_name}[{index}]") for index, channel in enumerate(value))


def _brush(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrokeSchemaError(f"{field_name} must be a non-empty string")
    return value

