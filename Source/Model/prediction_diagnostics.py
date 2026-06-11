"""Diagnostics for rendered BrushWright prediction exports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def compute_prediction_diagnostics(
    draft_path: Path,
    target_path: Path,
    predicted_path: Path,
    predicted_strokes: list[dict[str, Any]],
    target_strokes: list[dict[str, Any]] | None = None,
    min_changed_pixel_ratio: float = 0.005,
    min_gradient_improvement: float = 0.0,
    min_edge_overlap: float = 0.02,
) -> dict[str, Any]:
    draft_to_target = _image_delta(draft_path, target_path)
    draft_to_predicted = _image_delta(draft_path, predicted_path)
    predicted_to_target = _image_delta(predicted_path, target_path)
    visual_improved = predicted_to_target["mean_absolute_difference"] < draft_to_target["mean_absolute_difference"]
    changed_enough = draft_to_predicted["changed_pixel_ratio"] >= min_changed_pixel_ratio
    structure_metrics = _structure_metrics(draft_path, target_path, predicted_path)
    edge_improved = structure_metrics["edge_overlap"] >= min_edge_overlap
    gradient_improved = structure_metrics["gradient_improvement"] >= min_gradient_improvement
    material_masked_improvement = structure_metrics["masked_mad_improvement"] >= 0.005
    structure_improved = edge_improved and (gradient_improved or material_masked_improvement)
    if not changed_enough:
        status = "failed_low_pixel_change"
    elif not visual_improved:
        status = "failed_no_visual_improvement"
    elif not structure_improved:
        status = "failed_structure_noise"
    else:
        status = "improved"
    predicted_stats = stroke_statistics(predicted_strokes)
    target_stats = stroke_statistics(target_strokes) if target_strokes is not None else None
    return {
        "status": status,
        "visual_improved": visual_improved,
        "changed_enough": changed_enough,
        "structure_improved": structure_improved,
        "min_changed_pixel_ratio": min_changed_pixel_ratio,
        "min_gradient_improvement": min_gradient_improvement,
        "min_edge_overlap": min_edge_overlap,
        "image_deltas": {
            "draft_to_target": draft_to_target,
            "draft_to_predicted": draft_to_predicted,
            "predicted_to_target": predicted_to_target,
        },
        "structure_metrics": structure_metrics,
        "predicted_strokes": predicted_stats,
        "target_strokes": target_stats,
        "collapse_metrics": _collapse_metrics(predicted_stats, target_stats) if target_stats is not None else {},
    }


def stroke_statistics(strokes: list[dict[str, Any]]) -> dict[str, Any]:
    brushes = Counter(str(stroke.get("brush", "")) for stroke in strokes)
    stats: dict[str, Any] = {
        "count": len(strokes),
        "brush_histogram": dict(sorted(brushes.items())),
    }
    for field in ("x", "y", "angle", "length", "width", "opacity"):
        values = [float(stroke[field]) for stroke in strokes]
        stats[field] = _value_stats(values)
    colors = [stroke.get("color", [0.0, 0.0, 0.0]) for stroke in strokes]
    for index, name in enumerate(("r", "g", "b")):
        stats[name] = _value_stats([float(color[index]) for color in colors])
    return stats


def _image_delta(first_path: Path, second_path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for prediction diagnostics") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for prediction diagnostics") from exc

    with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
        first = first_image.convert("RGB")
        second = second_image.convert("RGB")
        if first.size != second.size:
            raise ValueError(f"image sizes differ: {first.size} != {second.size}")
        first_array = np.asarray(first, dtype=np.int16)
        second_array = np.asarray(second, dtype=np.int16)
    diff = np.abs(first_array - second_array)
    changed_pixels = diff.max(axis=2) > 2
    return {
        "mean_absolute_difference": float(diff.mean()),
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_pixel_ratio": float(changed_pixels.mean()),
    }


def _structure_metrics(draft_path: Path, target_path: Path, predicted_path: Path) -> dict[str, float]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for prediction diagnostics") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for prediction diagnostics") from exc

    with Image.open(draft_path) as draft_image, Image.open(target_path) as target_image, Image.open(predicted_path) as predicted_image:
        draft = np.asarray(draft_image.convert("RGB"), dtype=np.float32) / 255.0
        target = np.asarray(target_image.convert("RGB"), dtype=np.float32) / 255.0
        predicted = np.asarray(predicted_image.convert("RGB"), dtype=np.float32) / 255.0
    if draft.shape != target.shape or draft.shape != predicted.shape:
        raise ValueError("diagnostic image sizes differ")

    edit_mask = np.max(np.abs(target - draft), axis=2) > 0.04
    outside_mask = ~edit_mask
    draft_target = np.mean(np.abs(target - draft), axis=2)
    predicted_target = np.mean(np.abs(target - predicted), axis=2)
    masked_mad_improvement = float(
        _masked_mean(draft_target - predicted_target, edit_mask)
    )

    draft_grad = _sobel_magnitude_numpy(draft)
    target_grad = _sobel_magnitude_numpy(target)
    predicted_grad = _sobel_magnitude_numpy(predicted)
    draft_gradient_error = np.abs(target_grad - draft_grad)
    predicted_gradient_error = np.abs(target_grad - predicted_grad)
    gradient_improvement = float(_masked_mean(draft_gradient_error - predicted_gradient_error, edit_mask))

    target_delta_edges = _sobel_magnitude_numpy(np.abs(target - draft))
    predicted_delta_edges = _sobel_magnitude_numpy(np.abs(predicted - draft))
    target_edge_mask = (target_delta_edges > 0.08) & edit_mask
    predicted_edge_mask = (predicted_delta_edges > 0.08) & edit_mask
    target_edge_count = int(target_edge_mask.sum())
    edge_overlap = 1.0 if target_edge_count == 0 else float((target_edge_mask & predicted_edge_mask).sum() / target_edge_count)
    outside_mask_change = float(_masked_mean(np.mean(np.abs(predicted - draft), axis=2), outside_mask))
    return {
        "masked_mad_improvement": masked_mad_improvement,
        "gradient_improvement": gradient_improvement,
        "edge_overlap": edge_overlap,
        "target_edge_count": float(target_edge_count),
        "outside_mask_change": outside_mask_change,
    }


def _sobel_magnitude_numpy(image) -> Any:
    import numpy as np

    gray = image.mean(axis=2)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _masked_mean(values, mask) -> float:
    import numpy as np

    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def _value_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": variance ** 0.5,
    }


def _collapse_metrics(predicted_stats: dict[str, Any], target_stats: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for field in ("x", "y", "length", "width", "r", "g", "b"):
        predicted = predicted_stats[field]
        target = target_stats[field]
        target_std = max(float(target["std"]), 1e-8)
        metrics[field] = {
            "predicted_std": float(predicted["std"]),
            "target_std": float(target["std"]),
            "std_ratio": float(predicted["std"]) / target_std,
            "mean_delta": abs(float(predicted["mean"]) - float(target["mean"])),
        }
    return metrics
