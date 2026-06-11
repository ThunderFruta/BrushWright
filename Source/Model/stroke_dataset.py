"""PyTorch datasets for BrushWright stroke completion samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from Source.Model.stroke_tokenizer import StrokeTokenBatch, StrokeTokenizer


DEFAULT_CHUNK_SIZE = 64
DEFAULT_MAX_BASE_STROKES = 192
DEFAULT_IMAGE_SIZE = 512
V1_TARGET_CONTRACT = "paint_transformer_resplit_v1"
V1_BASE_STROKES = 192
V1_FINISHING_STROKES = 64


@dataclass(frozen=True)
class StrokeChunkIndex:
    sample_dir: Path
    sample_id: str
    chunk_start: int
    chunk_end: int
    stroke_count_adjusted: bool


@dataclass(frozen=True)
class StrokeDatasetItem:
    base_tokens: StrokeTokenBatch
    target_numeric: torch.Tensor
    target_brush_ids: torch.Tensor
    target_padding_mask: torch.Tensor
    draft_image: torch.Tensor
    goal_image: torch.Tensor | None
    error_map: torch.Tensor | None
    sample_id: str
    chunk_start: int
    chunk_end: int
    stroke_count_adjusted: bool


@dataclass(frozen=True)
class StrokeBatch:
    base_tokens: StrokeTokenBatch
    target_numeric: torch.Tensor
    target_brush_ids: torch.Tensor
    target_padding_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    chunk_starts: torch.Tensor
    chunk_ends: torch.Tensor
    stroke_count_adjusted: torch.Tensor
    draft_images: torch.Tensor | None = None
    goal_images: torch.Tensor | None = None
    error_maps: torch.Tensor | None = None


@dataclass(frozen=True)
class _CachedSample:
    base_tokens: StrokeTokenBatch
    draft_image: torch.Tensor
    goal_image: torch.Tensor | None
    error_map: torch.Tensor | None
    finishing_version: int
    finishing_canvas: dict[str, Any]
    finishing_metadata: dict[str, Any]
    finishing_strokes: list[dict[str, Any]]


class BrushWrightStrokeDataset(Dataset):
    """Expose finishing-stroke chunks from a BrushWright data split."""

    def __init__(
        self,
        split_root: Path | str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_base_strokes: int = DEFAULT_MAX_BASE_STROKES,
        tokenizer: StrokeTokenizer | None = None,
        cache_samples: bool = True,
        require_v1_contract: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.split_root = Path(split_root).resolve()
        self.chunk_size = chunk_size
        self.max_base_strokes = max_base_strokes
        self.cache_samples = cache_samples
        self.require_v1_contract = require_v1_contract
        self.tokenizer = tokenizer or StrokeTokenizer(max_strokes=max_base_strokes)
        self.target_tokenizer = StrokeTokenizer(
            max_strokes=chunk_size,
            brush_vocab=tuple(self.tokenizer.brush_to_id.keys()),
            device=self.tokenizer.device,
        )
        self.manifest = _read_json(self.split_root / "dataset_manifest.json")
        self.chunk_index = self._build_chunk_index()
        self._sample_cache: dict[Path, _CachedSample] = {}
        self._target_cache: dict[tuple[Path, int, int], StrokeTokenBatch] = {}

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, index: int) -> StrokeDatasetItem:
        chunk = self.chunk_index[index]
        cached_sample = self._load_cached_sample(chunk.sample_dir)
        target_tokens = self._load_cached_target(chunk, cached_sample)
        return StrokeDatasetItem(
            base_tokens=cached_sample.base_tokens,
            target_numeric=target_tokens.numeric[0],
            target_brush_ids=target_tokens.brush_ids[0],
            target_padding_mask=target_tokens.padding_mask[0],
            draft_image=cached_sample.draft_image,
            goal_image=cached_sample.goal_image,
            error_map=cached_sample.error_map,
            sample_id=chunk.sample_id,
            chunk_start=chunk.chunk_start,
            chunk_end=chunk.chunk_end,
            stroke_count_adjusted=chunk.stroke_count_adjusted,
        )

    def cache_stats(self) -> dict[str, int]:
        """Return per-process cache sizes for diagnostics and tests."""

        return {
            "samples": len(self._sample_cache),
            "targets": len(self._target_cache),
        }

    def _load_cached_sample(self, sample_dir: Path) -> _CachedSample:
        cached_sample = self._sample_cache.get(sample_dir)
        if cached_sample is not None:
            return cached_sample

        sample = _read_json(sample_dir / "sample.json")
        if self.require_v1_contract:
            _validate_v1_sample_contract(sample, sample_dir)
        base_program = _read_json(sample_dir / sample["base_strokes"])
        finishing_program = _read_json(sample_dir / sample["finishing_strokes"])
        draft_image = load_draft_image_tensor(sample_dir / sample["draft_image"])
        goal_image = _load_optional_image(sample_dir, sample.get("finished_image"))
        cached_sample = _CachedSample(
            base_tokens=self.tokenizer.encode_program(base_program),
            draft_image=draft_image,
            goal_image=goal_image,
            error_map=torch.abs(goal_image - draft_image) if goal_image is not None else None,
            finishing_version=finishing_program["version"],
            finishing_canvas=finishing_program["canvas"],
            finishing_metadata=dict(finishing_program.get("metadata", {})),
            finishing_strokes=finishing_program["strokes"],
        )
        if self.cache_samples:
            self._sample_cache[sample_dir] = cached_sample
        return cached_sample

    def _load_cached_target(self, chunk: StrokeChunkIndex, cached_sample: _CachedSample) -> StrokeTokenBatch:
        cache_key = (chunk.sample_dir, chunk.chunk_start, chunk.chunk_end)
        target_tokens = self._target_cache.get(cache_key)
        if target_tokens is not None:
            return target_tokens

        target_program = {
            "version": cached_sample.finishing_version,
            "canvas": cached_sample.finishing_canvas,
            "metadata": cached_sample.finishing_metadata,
            "strokes": cached_sample.finishing_strokes[chunk.chunk_start:chunk.chunk_end],
        }
        target_tokens = self.target_tokenizer.encode_program(target_program)
        if self.cache_samples:
            self._target_cache[cache_key] = target_tokens
        return target_tokens

    def _build_chunk_index(self) -> list[StrokeChunkIndex]:
        samples = self.manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"dataset manifest must contain a samples list: {self.split_root}")

        chunks: list[StrokeChunkIndex] = []
        for sample_entry in samples:
            sample_dir = self.split_root / sample_entry["path"]
            sample = _read_json(sample_dir / "sample.json")
            if self.require_v1_contract:
                _validate_v1_sample_contract(sample, sample_dir)
            finishing_count = int(sample["finishing_count"])
            if finishing_count <= 0:
                raise ValueError(f"sample has no finishing strokes: {sample_dir}")
            chunk_count = math.ceil(finishing_count / self.chunk_size)
            for chunk_index in range(chunk_count):
                chunk_start = chunk_index * self.chunk_size
                chunk_end = min(finishing_count, chunk_start + self.chunk_size)
                chunks.append(
                    StrokeChunkIndex(
                        sample_dir=sample_dir,
                        sample_id=str(sample["sample_id"]),
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        stroke_count_adjusted=bool(sample.get("stroke_count_adjusted", False)),
                    )
                )
        return chunks


def collate_stroke_chunks(items: Sequence[StrokeDatasetItem]) -> StrokeBatch:
    if not items:
        raise ValueError("items must contain at least one dataset item")

    base_tokens = StrokeTokenBatch(
        numeric=torch.cat([item.base_tokens.numeric for item in items], dim=0),
        brush_ids=torch.cat([item.base_tokens.brush_ids for item in items], dim=0),
        padding_mask=torch.cat([item.base_tokens.padding_mask for item in items], dim=0),
        lengths=torch.cat([item.base_tokens.lengths for item in items], dim=0),
    )
    return StrokeBatch(
        base_tokens=base_tokens,
        target_numeric=torch.stack([item.target_numeric for item in items], dim=0),
        target_brush_ids=torch.stack([item.target_brush_ids for item in items], dim=0),
        target_padding_mask=torch.stack([item.target_padding_mask for item in items], dim=0),
        sample_ids=tuple(item.sample_id for item in items),
        chunk_starts=torch.tensor([item.chunk_start for item in items], dtype=torch.long),
        chunk_ends=torch.tensor([item.chunk_end for item in items], dtype=torch.long),
        stroke_count_adjusted=torch.tensor([item.stroke_count_adjusted for item in items], dtype=torch.bool),
        draft_images=torch.stack([item.draft_image for item in items], dim=0),
        goal_images=_stack_optional([item.goal_image for item in items]),
        error_maps=_stack_optional([item.error_map for item in items]),
    )


def load_draft_image_tensor(path: Path | str, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Load a draft image as a normalized CHW tensor without adding torchvision."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load draft images") from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to load draft images") from exc

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize((image_size, image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_optional_image(sample_dir: Path, relative_path: str | None) -> torch.Tensor | None:
    if not relative_path:
        return None
    path = sample_dir / relative_path
    if not path.exists():
        return None
    return load_draft_image_tensor(path)


def _stack_optional(tensors: Sequence[torch.Tensor | None]) -> torch.Tensor | None:
    if any(tensor is None for tensor in tensors):
        return None
    return torch.stack([tensor for tensor in tensors if tensor is not None], dim=0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v1_sample_contract(sample: dict[str, Any], sample_dir: Path) -> None:
    expected = {
        "target_contract": V1_TARGET_CONTRACT,
        "base_count": V1_BASE_STROKES,
        "finishing_count": V1_FINISHING_STROKES,
        "render_draft_from_base": True,
        "draft_stroke_completion_delta": 0.0,
    }
    for key, expected_value in expected.items():
        if sample.get(key) != expected_value:
            raise ValueError(
                f"{sample_dir / 'sample.json'} must use V1 contract {key}={expected_value!r}; "
                f"got {sample.get(key)!r}"
            )
