"""Preprocess local image corpora for BrushWright sample generation.

The first preprocessing pass removes near-uniform borders from artwork images,
then writes normalized square images for Paint Transformer-backed synthesis.
Original source images are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_INPUT_DIR = Path("Assets/ImageCorpus/ArtInstituteChicago")
DEFAULT_OUTPUT_DIR = Path("Assets/ImageCorpus/ArtInstituteChicagoCropped")
DEFAULT_MANIFEST_PATH = Path("Outputs/ArtInstituteChicago/art_institute_chicago_cropped_manifest.json")
DEFAULT_CANVAS_SIZE = 512
DEFAULT_TOLERANCE = 18
DEFAULT_PADDING = 8
DEFAULT_CORNER_FRACTION = 0.08
DEFAULT_MIN_CONTENT_FRACTION = 0.08
BACKGROUND_COLOR = (255, 255, 255)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = preprocess_image_corpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            canvas_size=args.canvas_size,
            tolerance=args.tolerance,
            padding=args.padding,
            corner_fraction=args.corner_fraction,
            min_content_fraction=args.min_content_fraction,
            limit=args.limit,
            clear_existing=args.clear_existing,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"preprocess failed: {exc}")
        return 1
    print(f"Wrote {manifest['image_count']} preprocessed image(s) to {args.output_dir}")
    print(f"Manifest: {args.manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove near-uniform borders from a local image corpus.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    parser.add_argument("--corner-fraction", type=float, default=DEFAULT_CORNER_FRACTION)
    parser.add_argument("--min-content-fraction", type=float, default=DEFAULT_MIN_CONTENT_FRACTION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clear-existing", action="store_true")
    return parser


def preprocess_image_corpus(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    tolerance: int = DEFAULT_TOLERANCE,
    padding: int = DEFAULT_PADDING,
    corner_fraction: float = DEFAULT_CORNER_FRACTION,
    min_content_fraction: float = DEFAULT_MIN_CONTENT_FRACTION,
    limit: int | None = None,
    clear_existing: bool = False,
) -> dict[str, Any]:
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if not 0 < corner_fraction <= 0.25:
        raise ValueError("corner_fraction must be in (0, 0.25]")
    if not 0 < min_content_fraction <= 1:
        raise ValueError("min_content_fraction must be in (0, 1]")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not input_dir.exists():
        raise OSError(f"input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                path.unlink()

    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"no supported images found in {input_dir}")

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_dir / f"{image_path.stem}_cropped.jpg"
        with Image.open(image_path) as image:
            result = remove_uniform_border(
                image=image,
                canvas_size=canvas_size,
                tolerance=tolerance,
                padding=padding,
                corner_fraction=corner_fraction,
                min_content_fraction=min_content_fraction,
            )
            result.image.save(output_path, format="JPEG", quality=92, optimize=True)
        records.append(
            {
                "index": index - 1,
                "source_path": str(image_path),
                "path": str(output_path),
                "file_name": output_path.name,
                "original_width": result.original_size[0],
                "original_height": result.original_size[1],
                "content_bbox": list(result.content_bbox),
                "crop_bbox": list(result.crop_bbox),
                "cropped": result.cropped,
                "border_color": list(result.border_color),
                "width": canvas_size,
                "height": canvas_size,
            }
        )

    manifest = {
        "version": 1,
        "operation": "remove_uniform_border",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(records),
        "canvas_size": canvas_size,
        "tolerance": tolerance,
        "padding": padding,
        "corner_fraction": corner_fraction,
        "min_content_fraction": min_content_fraction,
        "images": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class BorderRemovalResult:
    def __init__(
        self,
        image: Image.Image,
        original_size: tuple[int, int],
        content_bbox: tuple[int, int, int, int],
        crop_bbox: tuple[int, int, int, int],
        cropped: bool,
        border_color: tuple[int, int, int],
    ) -> None:
        self.image = image
        self.original_size = original_size
        self.content_bbox = content_bbox
        self.crop_bbox = crop_bbox
        self.cropped = cropped
        self.border_color = border_color


def remove_uniform_border(
    image: Image.Image,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    tolerance: int = DEFAULT_TOLERANCE,
    padding: int = DEFAULT_PADDING,
    corner_fraction: float = DEFAULT_CORNER_FRACTION,
    min_content_fraction: float = DEFAULT_MIN_CONTENT_FRACTION,
) -> BorderRemovalResult:
    rgb = image.convert("RGB")
    width, height = rgb.size
    border_color = estimate_border_color(rgb, corner_fraction=corner_fraction)
    content_bbox = detect_content_bbox(
        rgb,
        border_color=border_color,
        tolerance=tolerance,
        min_content_fraction=min_content_fraction,
    )
    if content_bbox is None:
        content_bbox = (0, 0, width, height)
        crop_bbox = content_bbox
        cropped = False
    else:
        crop_bbox = expand_bbox(content_bbox, width=width, height=height, padding=padding)
        cropped = crop_bbox != (0, 0, width, height)

    cropped_image = rgb.crop(crop_bbox)
    normalized = normalize_to_square(cropped_image, canvas_size=canvas_size)
    return BorderRemovalResult(
        image=normalized,
        original_size=(width, height),
        content_bbox=content_bbox,
        crop_bbox=crop_bbox,
        cropped=cropped,
        border_color=border_color,
    )


def estimate_border_color(image: Image.Image, corner_fraction: float = DEFAULT_CORNER_FRACTION) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = arr.shape
    corner_w = max(1, int(width * corner_fraction))
    corner_h = max(1, int(height * corner_fraction))
    samples = np.concatenate(
        [
            arr[:corner_h, :corner_w].reshape(-1, 3),
            arr[:corner_h, -corner_w:].reshape(-1, 3),
            arr[-corner_h:, :corner_w].reshape(-1, 3),
            arr[-corner_h:, -corner_w:].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(samples, axis=0).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def detect_content_bbox(
    image: Image.Image,
    border_color: tuple[int, int, int],
    tolerance: int = DEFAULT_TOLERANCE,
    min_content_fraction: float = DEFAULT_MIN_CONTENT_FRACTION,
) -> tuple[int, int, int, int] | None:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.asarray(border_color, dtype=np.int16)
    delta = np.abs(arr - border).max(axis=2)
    mask = delta > tolerance
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    width, height = image.size
    content_area = (right - left) * (bottom - top)
    min_area = width * height * min_content_fraction
    if content_area < min_area:
        return None
    return left, top, right, bottom


def expand_bbox(bbox: tuple[int, int, int, int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def normalize_to_square(image: Image.Image, canvas_size: int = DEFAULT_CANVAS_SIZE) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    image = image.convert("RGB")
    image.thumbnail((canvas_size, canvas_size), resampling)
    canvas = Image.new("RGB", (canvas_size, canvas_size), BACKGROUND_COLOR)
    offset = ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
