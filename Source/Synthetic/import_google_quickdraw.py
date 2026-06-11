"""Import Google Quick, Draw! drawings into the local BrushWright image corpus.

Google Quick, Draw! provides millions of user doodles grouped by class. This
importer streams the simplified NDJSON class files, renders a fixed number of
vector drawings per class into square PNG images, and records a reproducibility
manifest. Local imported images are ignored source-image artifacts.
"""

from __future__ import annotations

import argparse
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw


DEFAULT_CATEGORIES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DEFAULT_SIMPLIFIED_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
DEFAULT_PER_CLASS = 200
DEFAULT_OUTPUT_DIR = Path("Assets/ImageCorpus/GoogleQuickDraw")
DEFAULT_MANIFEST_PATH = Path("Outputs/GoogleQuickDraw/google_quickdraw_import_manifest.json")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_STROKE_WIDTH = 5
IMAGE_SUFFIX = ".png"
USER_AGENT = "BrushWright/0.1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    socket.setdefaulttimeout(args.timeout_seconds)

    try:
        classes = _resolve_classes(args)
        result = import_google_quickdraw(
            classes=classes,
            per_class=args.per_class,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            image_size=args.image_size,
            stroke_width=args.stroke_width,
            simplified_base_url=args.simplified_base_url,
            recognized_only=not args.include_unrecognized,
            clear_existing=args.clear_existing,
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"google quickdraw import failed: {exc}", flush=True)
        return 1

    print(f"Wrote {result['image_count']} Google Quick, Draw! image(s) to {args.output_dir}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Google Quick, Draw! drawings into Assets/ImageCorpus/GoogleQuickDraw.")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Class name to import. Repeatable. Default: all Quick, Draw! classes.")
    parser.add_argument("--classes-file", type=Path, default=None, help="Optional newline-delimited class list.")
    parser.add_argument("--categories-url", default=DEFAULT_CATEGORIES_URL)
    parser.add_argument("--simplified-base-url", default=DEFAULT_SIMPLIFIED_BASE_URL)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--max-classes", type=int, default=None, help="Optional cap for smoke tests or partial imports.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--stroke-width", type=int, default=DEFAULT_STROKE_WIDTH)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--include-unrecognized", action="store_true", help="Include records marked recognized=false.")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing PNGs before writing the imported corpus.")
    return parser


def _resolve_classes(args: argparse.Namespace) -> list[str]:
    classes: list[str] = []
    if args.classes:
        classes.extend(args.classes)
    if args.classes_file is not None:
        classes.extend(_class_lines(args.classes_file.read_text(encoding="utf-8")))
    if not classes:
        classes = fetch_quickdraw_classes(args.categories_url)
    classes = _dedupe_preserve_order(classes)
    if args.max_classes is not None:
        if args.max_classes <= 0:
            raise ValueError("max_classes must be positive")
        classes = classes[: args.max_classes]
    if not classes:
        raise ValueError("no Quick, Draw! classes selected")
    return classes


def fetch_quickdraw_classes(categories_url: str = DEFAULT_CATEGORIES_URL) -> list[str]:
    request = urllib.request.Request(categories_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return _class_lines(response.read().decode("utf-8"))


def import_google_quickdraw(
    classes: Sequence[str],
    per_class: int = DEFAULT_PER_CLASS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    stroke_width: int = DEFAULT_STROKE_WIDTH,
    simplified_base_url: str = DEFAULT_SIMPLIFIED_BASE_URL,
    recognized_only: bool = True,
    clear_existing: bool = False,
) -> dict[str, Any]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if stroke_width <= 0:
        raise ValueError("stroke_width must be positive")
    if not classes:
        raise ValueError("classes must not be empty")

    output_dir = output_dir.expanduser()
    manifest_path = manifest_path.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for path in output_dir.glob(f"*{IMAGE_SUFFIX}"):
            path.unlink()

    images: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    class_summaries: list[dict[str, Any]] = []
    exported_total = 0
    selected_classes = _dedupe_preserve_order(classes)

    for class_index, class_name in enumerate(selected_classes, start=1):
        exported_for_class = 0
        scanned_for_class = 0
        class_slug = _slug(class_name)
        class_url = quickdraw_class_url(class_name, simplified_base_url)
        print(f"[{class_index}/{len(selected_classes)}] {class_name}: importing {per_class}", flush=True)
        try:
            records = iter_quickdraw_records(class_url)
            for record in records:
                scanned_for_class += 1
                if recognized_only and not bool(record.get("recognized", False)):
                    continue
                drawing = record.get("drawing")
                if not _valid_drawing(drawing):
                    continue
                file_name = f"quickdraw_{class_slug}_{exported_for_class + 1:04d}{IMAGE_SUFFIX}"
                output_path = output_dir / file_name
                render_quickdraw_drawing(
                    drawing=drawing,
                    output_path=output_path,
                    image_size=image_size,
                    stroke_width=stroke_width,
                )
                images.append(
                    {
                        "index": exported_total,
                        "class_index": class_index - 1,
                        "class_name": class_name,
                        "class_slug": class_slug,
                        "class_item_index": exported_for_class,
                        "path": str(output_path),
                        "file_name": file_name,
                        "key_id": str(record.get("key_id") or ""),
                        "countrycode": str(record.get("countrycode") or ""),
                        "recognized": bool(record.get("recognized", False)),
                        "source_url": class_url,
                        "width": image_size,
                        "height": image_size,
                    }
                )
                exported_total += 1
                exported_for_class += 1
                if exported_for_class >= per_class:
                    break
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            failed.append({"class_name": class_name, "error": str(exc)})

        class_summaries.append(
            {
                "class_name": class_name,
                "class_slug": class_slug,
                "source_url": class_url,
                "requested": per_class,
                "exported": exported_for_class,
                "scanned": scanned_for_class,
            }
        )
        print(f"[{class_index}/{len(selected_classes)}] {class_name}: wrote {exported_for_class}/{per_class}", flush=True)

    if exported_total == 0:
        raise ValueError("no Google Quick, Draw! images were imported")

    manifest = {
        "version": 1,
        "source_dataset": "Google Quick, Draw!",
        "source_reference": "https://quickdraw.withgoogle.com/data",
        "source_access": "Simplified NDJSON files from Google Cloud Storage",
        "simplified_base_url": simplified_base_url,
        "license": "CC-BY-4.0",
        "license_note": "Google Quick, Draw! dataset drawings are made available by Google under Creative Commons Attribution 4.0 International.",
        "output_dir": str(output_dir),
        "class_count": len(selected_classes),
        "per_class": per_class,
        "image_count": exported_total,
        "image_size": image_size,
        "stroke_width": stroke_width,
        "recognized_only": recognized_only,
        "failed_count": len(failed),
        "classes": class_summaries,
        "images": images,
        "failed": failed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def quickdraw_class_url(class_name: str, simplified_base_url: str = DEFAULT_SIMPLIFIED_BASE_URL) -> str:
    encoded = urllib.parse.quote(class_name, safe="")
    return f"{simplified_base_url.rstrip('/')}/{encoded}.ndjson"


def iter_quickdraw_records(url: str) -> Iterable[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def render_quickdraw_drawing(
    drawing: Any,
    output_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    stroke_width: int = DEFAULT_STROKE_WIDTH,
) -> None:
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    scale = image_size / 255.0
    width = max(1, round(stroke_width * image_size / DEFAULT_IMAGE_SIZE))
    for stroke in drawing:
        xs, ys = stroke
        if len(xs) != len(ys) or not xs:
            continue
        points = [(float(x) * scale, float(y) * scale) for x, y in zip(xs, ys)]
        if len(points) == 1:
            x, y = points[0]
            radius = max(1, width // 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")
        else:
            draw.line(points, fill="black", width=width, joint="curve")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _valid_drawing(drawing: Any) -> bool:
    if not isinstance(drawing, list) or not drawing:
        return False
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) != 2:
            return False
        xs, ys = stroke
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            return False
    return True


def _class_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "quickdraw"


if __name__ == "__main__":
    raise SystemExit(main())
