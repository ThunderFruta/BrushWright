"""Patch datasets for visual-delta-to-stroke training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from Source.Model.stroke_dataset import DEFAULT_IMAGE_SIZE, load_draft_image_tensor
from Source.Model.stroke_tokenizer import NUMERIC_FIELDS, StrokeTokenizer


DEFAULT_PATCH_SIZE = 64
DEFAULT_PATCH_STRIDE = 64
DEFAULT_MAX_STROKES_PER_PATCH = 256
DEFAULT_MASK_THRESHOLD = 0.04
DEFAULT_MIN_CHANGED_PIXELS = 16
DEFAULT_NEGATIVE_PATCH_RATIO = 0.25
DEFAULT_EDGE_FOCUSED_SAMPLING = True
STRUCTURE_TARGET_SELECTION_MODE = "structure_first_v1"
TARGET_CONTRACT_ORIGINAL_IMAGE_TARGET = "paint_transformer_original_image_target_v1"
TARGET_CONTRACT_OUTPUT_DETAIL_PAIR = "paint_transformer_output_detail_pair_v1"


@dataclass(frozen=True)
class VisualDeltaPatchIndex:
    sample_dir: Path
    sample_id: str
    left: int
    top: int
    patch_size: int
    changed: bool
    target_stroke_count: int = 0
    edge_density: float = 0.0
    error_score: float = 0.0
    priority_score: float = 0.0


@dataclass(frozen=True)
class VisualDeltaDatasetItem:
    patch_tensor: torch.Tensor
    target_numeric: torch.Tensor
    target_brush_ids: torch.Tensor
    target_present: torch.Tensor
    target_padding_mask: torch.Tensor
    sample_id: str
    patch_bounds: torch.Tensor
    changed: bool


@dataclass(frozen=True)
class VisualDeltaBatch:
    patch_tensor: torch.Tensor
    target_numeric: torch.Tensor
    target_brush_ids: torch.Tensor
    target_present: torch.Tensor
    target_padding_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    patch_bounds: torch.Tensor
    changed: torch.Tensor


@dataclass(frozen=True)
class _CachedVisualDeltaSample:
    draft_image: torch.Tensor
    target_image: torch.Tensor
    error_map: torch.Tensor
    edit_mask: torch.Tensor
    target_strokes: list[dict[str, Any]]


class VisualDeltaStrokeDataset(Dataset):
    """Expose image-delta patches with local finishing-stroke targets."""

    def __init__(
        self,
        split_root: Path | str,
        patch_size: int = DEFAULT_PATCH_SIZE,
        patch_stride: int = DEFAULT_PATCH_STRIDE,
        max_strokes_per_patch: int = DEFAULT_MAX_STROKES_PER_PATCH,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        min_changed_pixels: int = DEFAULT_MIN_CHANGED_PIXELS,
        negative_patch_ratio: float = DEFAULT_NEGATIVE_PATCH_RATIO,
        edge_focused_sampling: bool = DEFAULT_EDGE_FOCUSED_SAMPLING,
        tokenizer: StrokeTokenizer | None = None,
        cache_samples: bool = True,
        require_structure_targets: bool = False,
        require_target_contract: str | None = None,
        include_zero_target_changed_patches: bool = True,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if max_strokes_per_patch <= 0:
            raise ValueError("max_strokes_per_patch must be positive")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if min_changed_pixels < 0:
            raise ValueError("min_changed_pixels must be non-negative")
        if negative_patch_ratio < 0.0:
            raise ValueError("negative_patch_ratio must be non-negative")
        self.split_root = Path(split_root).resolve()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.max_strokes_per_patch = max_strokes_per_patch
        self.mask_threshold = mask_threshold
        self.min_changed_pixels = min_changed_pixels
        self.negative_patch_ratio = negative_patch_ratio
        self.edge_focused_sampling = edge_focused_sampling
        self.cache_samples = cache_samples
        self.require_structure_targets = require_structure_targets
        self.require_target_contract = require_target_contract
        self.include_zero_target_changed_patches = include_zero_target_changed_patches
        self.tokenizer = tokenizer or StrokeTokenizer(max_strokes=max_strokes_per_patch)
        self.manifest = _read_json(self.split_root / "dataset_manifest.json")
        self.patch_index = self._build_patch_index()
        self._sample_cache: dict[Path, _CachedVisualDeltaSample] = {}

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, index: int) -> VisualDeltaDatasetItem:
        patch = self.patch_index[index]
        sample = self._load_cached_sample(patch.sample_dir)
        left = patch.left
        top = patch.top
        right = left + patch.patch_size
        bottom = top + patch.patch_size
        draft_patch = sample.draft_image[:, top:bottom, left:right]
        target_patch = sample.target_image[:, top:bottom, left:right]
        error_patch = sample.error_map[:, top:bottom, left:right]
        mask_patch = sample.edit_mask[:, top:bottom, left:right]
        patch_tensor = torch.cat([draft_patch, target_patch, error_patch, mask_patch], dim=0)
        target_numeric, target_brush_ids, target_present, target_padding_mask = self._target_tensors(
            sample.target_strokes,
            left=left,
            top=top,
            patch_size=patch.patch_size,
        )
        return VisualDeltaDatasetItem(
            patch_tensor=patch_tensor,
            target_numeric=target_numeric,
            target_brush_ids=target_brush_ids,
            target_present=target_present,
            target_padding_mask=target_padding_mask,
            sample_id=patch.sample_id,
            patch_bounds=torch.tensor([left, top, right, bottom], dtype=torch.float32) / DEFAULT_IMAGE_SIZE,
            changed=patch.changed,
        )

    def cache_stats(self) -> dict[str, int]:
        return {"samples": len(self._sample_cache)}

    def _build_patch_index(self) -> list[VisualDeltaPatchIndex]:
        samples = self.manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"dataset manifest must contain a samples list: {self.split_root}")
        patch_index: list[VisualDeltaPatchIndex] = []
        for sample_entry in samples:
            sample_dir = self.split_root / sample_entry["path"]
            sample = _read_json(sample_dir / "sample.json")
            self._validate_target_contract_sample(sample_dir, sample)
            self._validate_structure_target_sample(sample_dir, sample)
            draft_image = load_draft_image_tensor(sample_dir / sample["draft_image"])
            target_image = load_draft_image_tensor(_target_image_path(sample_dir, sample))
            target_program = _read_json(_target_strokes_path(sample_dir, sample))
            target_strokes = target_program["strokes"]
            error_map = torch.abs(target_image - draft_image)
            edit_mask = _build_edit_mask(error_map, self.mask_threshold)
            edge_map = _build_structure_edge_map(error_map)
            changed: list[VisualDeltaPatchIndex] = []
            unchanged: list[VisualDeltaPatchIndex] = []
            for top in _patch_offsets(DEFAULT_IMAGE_SIZE, self.patch_size, self.patch_stride):
                for left in _patch_offsets(DEFAULT_IMAGE_SIZE, self.patch_size, self.patch_stride):
                    mask_patch = edit_mask[:, top : top + self.patch_size, left : left + self.patch_size]
                    edge_patch = edge_map[:, top : top + self.patch_size, left : left + self.patch_size]
                    error_patch = error_map[:, top : top + self.patch_size, left : left + self.patch_size]
                    is_changed = int(mask_patch.sum().item()) >= self.min_changed_pixels
                    target_stroke_count = sum(
                        1
                        for stroke in target_strokes
                        if _stroke_inside_patch(stroke, left=left, top=top, patch_size=self.patch_size)
                    )
                    edge_density = float((edge_patch * mask_patch).mean().item())
                    error_score = float((error_patch.max(dim=0, keepdim=True).values * mask_patch).mean().item())
                    priority_score = _patch_priority_score(
                        target_stroke_count=target_stroke_count,
                        edge_density=edge_density,
                        error_score=error_score,
                        edge_focused_sampling=self.edge_focused_sampling,
                    )
                    patch = VisualDeltaPatchIndex(
                        sample_dir=sample_dir,
                        sample_id=str(sample["sample_id"]),
                        left=left,
                        top=top,
                        patch_size=self.patch_size,
                        changed=is_changed,
                        target_stroke_count=target_stroke_count,
                        edge_density=edge_density,
                        error_score=error_score,
                        priority_score=priority_score,
                    )
                    if is_changed:
                        if target_stroke_count > 0 or self.include_zero_target_changed_patches:
                            changed.append(patch)
                    else:
                        unchanged.append(patch)
            if changed:
                changed.sort(key=lambda patch: (patch.priority_score, patch.target_stroke_count, -patch.top, -patch.left), reverse=True)
                patch_index.extend(changed)
                negative_count = min(len(unchanged), max(1, int(len(changed) * self.negative_patch_ratio)))
                patch_index.extend(unchanged[:negative_count])
            else:
                patch_index.extend(unchanged[:1])
        return patch_index

    def _load_cached_sample(self, sample_dir: Path) -> _CachedVisualDeltaSample:
        cached = self._sample_cache.get(sample_dir)
        if cached is not None:
            return cached
        sample = _read_json(sample_dir / "sample.json")
        self._validate_structure_target_sample(sample_dir, sample)
        draft_image = load_draft_image_tensor(sample_dir / sample["draft_image"])
        target_image = load_draft_image_tensor(_target_image_path(sample_dir, sample))
        error_map = torch.abs(target_image - draft_image)
        edit_mask = _build_edit_mask(error_map, self.mask_threshold)
        target_program = _read_json(_target_strokes_path(sample_dir, sample))
        cached = _CachedVisualDeltaSample(
            draft_image=draft_image,
            target_image=target_image,
            error_map=error_map,
            edit_mask=edit_mask,
            target_strokes=target_program["strokes"],
        )
        if self.cache_samples:
            self._sample_cache[sample_dir] = cached
        return cached

    def _target_tensors(
        self,
        strokes: Sequence[dict[str, Any]],
        left: int,
        top: int,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        numeric = torch.zeros(self.max_strokes_per_patch, len(NUMERIC_FIELDS), dtype=torch.float32)
        brush_ids = torch.zeros(self.max_strokes_per_patch, dtype=torch.long)
        present = torch.zeros(self.max_strokes_per_patch, dtype=torch.float32)
        padding_mask = torch.ones(self.max_strokes_per_patch, dtype=torch.bool)
        selected = [
            stroke
            for stroke in strokes
            if _stroke_inside_patch(stroke, left=left, top=top, patch_size=patch_size)
        ][: self.max_strokes_per_patch]
        for index, stroke in enumerate(selected):
            local = _stroke_to_patch_numeric(stroke, left=left, top=top, patch_size=patch_size)
            numeric[index] = torch.tensor(local, dtype=torch.float32)
            brush_ids[index] = self.tokenizer.brush_to_id.get(str(stroke.get("brush", "")), 1)
            present[index] = 1.0
            padding_mask[index] = False
        return numeric, brush_ids, present, padding_mask

    def _validate_structure_target_sample(self, sample_dir: Path, sample: dict[str, Any]) -> None:
        if not self.require_structure_targets:
            return
        mode = sample.get("target_selection_mode")
        if mode != STRUCTURE_TARGET_SELECTION_MODE:
            raise ValueError(
                f"{sample_dir} target_selection_mode must be {STRUCTURE_TARGET_SELECTION_MODE!r}; got {mode!r}"
            )
        manifest_name = sample.get("target_selection_manifest")
        if not manifest_name:
            raise ValueError(f"{sample_dir} is missing target_selection_manifest")
        manifest_path = sample_dir / str(manifest_name)
        if not manifest_path.exists():
            raise ValueError(f"{sample_dir} target_selection_manifest does not exist: {manifest_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("target_selection_mode") != STRUCTURE_TARGET_SELECTION_MODE:
            raise ValueError(
                f"{manifest_path} target_selection_mode must be {STRUCTURE_TARGET_SELECTION_MODE!r}"
            )

    def _validate_target_contract_sample(self, sample_dir: Path, sample: dict[str, Any]) -> None:
        if not self.require_target_contract:
            return
        contract = sample.get("target_contract")
        if contract != self.require_target_contract:
            raise ValueError(
                f"{sample_dir} target_contract must be {self.require_target_contract!r}; got {contract!r}"
            )
        split_manifest_name = sample.get("split_manifest")
        if not split_manifest_name:
            raise ValueError(f"{sample_dir} is missing split_manifest")
        split_manifest_path = sample_dir / str(split_manifest_name)
        if not split_manifest_path.exists():
            raise ValueError(f"{sample_dir} split_manifest does not exist: {split_manifest_path}")
        split_manifest = _read_json(split_manifest_path)
        if split_manifest.get("target_contract") != self.require_target_contract:
            raise ValueError(
                f"{split_manifest_path} target_contract must be {self.require_target_contract!r}"
            )


def collate_visual_delta_patches(items: Sequence[VisualDeltaDatasetItem]) -> VisualDeltaBatch:
    if not items:
        raise ValueError("items must contain at least one dataset item")
    return VisualDeltaBatch(
        patch_tensor=torch.stack([item.patch_tensor for item in items], dim=0),
        target_numeric=torch.stack([item.target_numeric for item in items], dim=0),
        target_brush_ids=torch.stack([item.target_brush_ids for item in items], dim=0),
        target_present=torch.stack([item.target_present for item in items], dim=0),
        target_padding_mask=torch.stack([item.target_padding_mask for item in items], dim=0),
        sample_ids=tuple(item.sample_id for item in items),
        patch_bounds=torch.stack([item.patch_bounds for item in items], dim=0),
        changed=torch.tensor([item.changed for item in items], dtype=torch.bool),
    )


def visual_delta_batch_to_device(batch: VisualDeltaBatch, device: torch.device) -> VisualDeltaBatch:
    non_blocking = device.type == "cuda"
    return VisualDeltaBatch(
        patch_tensor=batch.patch_tensor.to(device, non_blocking=non_blocking),
        target_numeric=batch.target_numeric.to(device, non_blocking=non_blocking),
        target_brush_ids=batch.target_brush_ids.to(device, non_blocking=non_blocking),
        target_present=batch.target_present.to(device, non_blocking=non_blocking),
        target_padding_mask=batch.target_padding_mask.to(device, non_blocking=non_blocking),
        sample_ids=batch.sample_ids,
        patch_bounds=batch.patch_bounds.to(device, non_blocking=non_blocking),
        changed=batch.changed.to(device, non_blocking=non_blocking),
    )


def patch_numeric_to_global_stroke(values: Sequence[float], brush: str, patch_bounds: Sequence[float]) -> dict[str, Any]:
    left, top, right, bottom = [float(value) for value in patch_bounds]
    patch_width = right - left
    patch_height = bottom - top
    clipped = [max(0.0, min(1.0, float(value))) for value in values]
    return {
        "x": max(0.0, min(1.0, left + clipped[0] * patch_width)),
        "y": max(0.0, min(1.0, top + clipped[1] * patch_height)),
        "angle": clipped[2],
        "length": max(0.0, min(1.0, clipped[3] * patch_width)),
        "width": max(0.0, min(1.0, clipped[4] * patch_width)),
        "opacity": clipped[5],
        "color": [clipped[6], clipped[7], clipped[8]],
        "brush": brush,
    }


def _build_edit_mask(error_map: torch.Tensor, threshold: float) -> torch.Tensor:
    return (error_map.max(dim=0, keepdim=True).values > threshold).to(torch.float32)


def _build_structure_edge_map(error_map: torch.Tensor) -> torch.Tensor:
    gray = error_map.mean(dim=0, keepdim=True)
    horizontal = torch.zeros_like(gray)
    vertical = torch.zeros_like(gray)
    horizontal[:, :, 1:] = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
    vertical[:, 1:, :] = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
    return torch.clamp(horizontal + vertical, 0.0, 1.0)


def _patch_priority_score(
    target_stroke_count: int,
    edge_density: float,
    error_score: float,
    edge_focused_sampling: bool,
) -> float:
    if not edge_focused_sampling:
        return float(target_stroke_count)
    return float(target_stroke_count) + edge_density * 200.0 + error_score * 20.0


def _patch_offsets(image_size: int, patch_size: int, patch_stride: int) -> list[int]:
    if patch_size > image_size:
        raise ValueError("patch_size must not exceed image_size")
    offsets = list(range(0, image_size - patch_size + 1, patch_stride))
    final = image_size - patch_size
    if offsets[-1] != final:
        offsets.append(final)
    return offsets


def _stroke_inside_patch(stroke: dict[str, Any], left: int, top: int, patch_size: int) -> bool:
    x = float(stroke["x"]) * DEFAULT_IMAGE_SIZE
    y = float(stroke["y"]) * DEFAULT_IMAGE_SIZE
    return left <= x < left + patch_size and top <= y < top + patch_size


def _stroke_to_patch_numeric(stroke: dict[str, Any], left: int, top: int, patch_size: int) -> list[float]:
    scale = DEFAULT_IMAGE_SIZE / patch_size
    color = stroke.get("color", [0.0, 0.0, 0.0])
    return [
        _clamp01((float(stroke["x"]) * DEFAULT_IMAGE_SIZE - left) / patch_size),
        _clamp01((float(stroke["y"]) * DEFAULT_IMAGE_SIZE - top) / patch_size),
        _clamp01(float(stroke["angle"])),
        _clamp01(float(stroke["length"]) * scale),
        _clamp01(float(stroke["width"]) * scale),
        _clamp01(float(stroke["opacity"])),
        _clamp01(float(color[0])),
        _clamp01(float(color[1])),
        _clamp01(float(color[2])),
    ]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_image_path(sample_dir: Path, sample: dict[str, Any]) -> Path:
    image_name = sample.get("target_image") or sample.get("finished_image")
    if not image_name:
        raise ValueError(f"{sample_dir} sample is missing target_image metadata")
    return sample_dir / str(image_name)


def _target_strokes_path(sample_dir: Path, sample: dict[str, Any]) -> Path:
    strokes_name = sample.get("visual_teacher_strokes") or sample.get("finishing_strokes")
    if not strokes_name:
        raise ValueError(f"{sample_dir} sample is missing finishing_strokes metadata")
    path = sample_dir / str(strokes_name)
    if not path.exists():
        raise ValueError(f"{sample_dir} target stroke program does not exist: {path}")
    return path
