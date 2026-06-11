"""Minimal image metrics for pre-ML renderer and dataset checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageStat


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two rendered images with basic metrics.")
    parser.add_argument("actual", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    metrics = compare_images(args.actual, args.expected)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, indent=2)
            output_file.write("\n")
    else:
        print(json.dumps(metrics, indent=2))
    return 0


def compare_images(actual_path: Path, expected_path: Path) -> dict[str, object]:
    with Image.open(actual_path) as actual_image, Image.open(expected_path) as expected_image:
        actual = actual_image.convert("RGB")
        expected = expected_image.convert("RGB")
        if actual.size != expected.size:
            raise ValueError(f"image sizes differ: {actual.size} != {expected.size}")

        diff = ImageChops.difference(actual, expected)
        stat = ImageStat.Stat(diff)
        pixel_count = actual.size[0] * actual.size[1]
        channel_mse = [channel_sum2 / pixel_count for channel_sum2 in stat.sum2]
        mse = sum(channel_mse) / 3.0
        mean_abs_diff = sum(stat.mean) / 3.0

        return {
            "actual": str(actual_path),
            "expected": str(expected_path),
            "width": actual.size[0],
            "height": actual.size[1],
            "pixel_mse": mse,
            "mean_absolute_difference": mean_abs_diff,
        }


if __name__ == "__main__":
    raise SystemExit(main())
