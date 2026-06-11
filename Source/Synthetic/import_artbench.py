"""Import ArtBench images into the local BrushWright image corpus.

ArtBench is kept as an external, opt-in source image corpus. This module does
not vendor the dataset; it samples rows through Hugging Face datasets and writes
local image files under Assets/ImageCorpus/ArtBench by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET_NAME = "zguo0525/ArtBench"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 64
DEFAULT_OUTPUT_DIR = Path("Assets/ImageCorpus/ArtBench")
DEFAULT_MANIFEST_PATH = Path("Outputs/ArtBench/artbench_import_manifest.json")
DEFAULT_SEED = 1
DEFAULT_SHUFFLE_BUFFER = 10_000
IMAGE_SUFFIX = ".png"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        rows, label_names = _load_artbench_rows(
            dataset_name=args.dataset_name,
            split=args.split,
            streaming=not args.no_streaming,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        result = export_artbench_rows(
            rows=rows,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            styles=args.style,
            label_names=label_names,
            image_size=args.image_size,
            source_dataset=args.dataset_name,
            split=args.split,
            seed=args.seed,
            clear_existing=args.clear_existing,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"artbench import failed: {exc}")
        return 1

    print(f"Wrote {result['image_count']} ArtBench image(s) to {args.output_dir}")
    print(f"Manifest: {args.manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import ArtBench images into Assets/ImageCorpus/ArtBench.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--style", action="append", default=[], help="Optional style label filter. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square resize before saving. Default preserves dataset image size.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep source dataset order.")
    parser.add_argument("--no-streaming", action="store_true", help="Download/load the split locally before export.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing PNGs in the output directory before writing the sampled corpus.",
    )
    return parser


def _load_artbench_rows(
    dataset_name: str,
    split: str,
    streaming: bool,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for ArtBench import. Install it with: "
            "python3 -m pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    label_names = _label_names_from_features(getattr(dataset, "features", None))
    if shuffle:
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=seed)
    return dataset, label_names


def export_artbench_rows(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    limit: int,
    styles: Sequence[str],
    label_names: Sequence[str] | None = None,
    image_size: int | None = None,
    source_dataset: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    clear_existing: bool = False,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if image_size is not None and image_size <= 0:
        raise ValueError("image_size must be positive")

    output_dir = output_dir.expanduser()
    manifest_path = manifest_path.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for existing_image in output_dir.glob(f"*{IMAGE_SUFFIX}"):
            existing_image.unlink()

    style_filters = {_slug(style) for style in styles}
    images: list[dict[str, Any]] = []
    style_counts: dict[str, int] = {}
    exported = 0

    for row_index, row in enumerate(rows):
        label = _row_label(row, label_names)
        label_slug = _slug(label)
        if style_filters and label_slug not in style_filters:
            continue

        image_value = row.get("image")
        if image_value is None:
            continue

        style_counts[label_slug] = style_counts.get(label_slug, 0) + 1
        file_name = f"artbench_{exported + 1:06d}_{label_slug}{IMAGE_SUFFIX}"
        output_path = output_dir / file_name
        width, height = _save_image(image_value, output_path, image_size=image_size)
        images.append(
            {
                "index": exported,
                "source_row_index": row_index,
                "path": str(output_path),
                "file_name": file_name,
                "label": label,
                "label_slug": label_slug,
                "width": width,
                "height": height,
            }
        )
        exported += 1
        if exported >= limit:
            break

    if exported == 0:
        style_hint = f" matching styles {sorted(style_filters)}" if style_filters else ""
        raise ValueError(f"no ArtBench images found{style_hint}")

    manifest = {
        "version": 1,
        "source_dataset": source_dataset,
        "source_reference": "https://github.com/liaopeiyuan/artbench",
        "source_access": "Hugging Face datasets mirror",
        "license_note": "ArtBench is distributed for fair-use research; keep imported images external to Git.",
        "split": split,
        "seed": seed,
        "limit": limit,
        "style_filters": sorted(style_filters),
        "output_dir": str(output_dir),
        "image_count": exported,
        "style_counts": style_counts,
        "images": images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _label_names_from_features(features: Any) -> Sequence[str] | None:
    if features is None:
        return None
    try:
        label_feature = features.get("label")
    except AttributeError:
        return None
    names = getattr(label_feature, "names", None)
    return tuple(names) if names else None


def _row_label(row: dict[str, Any], label_names: Sequence[str] | None) -> str:
    if "label" in row:
        return _label_to_name(row["label"], label_names)
    if "prompt" in row:
        return _label_from_prompt(str(row["prompt"]))
    if "style" in row:
        return str(row["style"])
    return "unknown"


def _label_to_name(label: Any, label_names: Sequence[str] | None) -> str:
    if isinstance(label, str):
        return label
    if isinstance(label, int) and label_names is not None and 0 <= label < len(label_names):
        return str(label_names[label])
    return str(label)


def _label_from_prompt(prompt: str) -> str:
    normalized = prompt.strip().lower()
    normalized = re.sub(r"^an? ", "", normalized)
    normalized = re.sub(r" painting$", "", normalized)
    return normalized or "unknown"


def _save_image(image_value: Any, output_path: Path, image_size: int | None) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export ArtBench images") from exc

    if hasattr(image_value, "convert"):
        image = image_value.convert("RGB")
    elif isinstance(image_value, (str, Path)):
        image = Image.open(image_value).convert("RGB")
    elif isinstance(image_value, dict) and image_value.get("bytes") is not None:
        from io import BytesIO

        image = Image.open(BytesIO(image_value["bytes"])).convert("RGB")
    elif isinstance(image_value, dict) and image_value.get("path") is not None:
        image = Image.open(image_value["path"]).convert("RGB")
    else:
        raise ValueError(f"unsupported image value: {type(image_value).__name__}")

    if image_size is not None:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((image_size, image_size), resampling)
    image.save(output_path)
    return image.size


def _slug(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def shuffled_subset(paths: Sequence[Path], limit: int, seed: int = DEFAULT_SEED) -> list[Path]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    rng = random.Random(seed)
    selected = list(paths)
    rng.shuffle(selected)
    return selected[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
