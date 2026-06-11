"""Import Openclipart images into the local BrushWright image corpus.

Openclipart is the preferred broad image corpus for BrushWright because its
artwork is published under CC0/public-domain terms. This importer downloads PNG
renditions through Openclipart's JSON search API and keeps files as local,
ignored source-image artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_API_URL = "https://openclipart.org/search/json/"
FALLBACK_API_URL = "http://openclipart.org/search/json/"
DEFAULT_QUERIES = ("tree", "flower", "house", "face", "animal", "landscape", "object", "icon")
DEFAULT_LIMIT = 64
DEFAULT_PER_QUERY = 24
DEFAULT_OUTPUT_DIR = Path("Assets/ImageCorpus/Openclipart")
DEFAULT_MANIFEST_PATH = Path("Outputs/Openclipart/openclipart_import_manifest.json")
DEFAULT_IMAGE_SIZE = 512
IMAGE_SUFFIX = ".png"
USER_AGENT = "BrushWright/0.1 (+https://github.com/)"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    queries = tuple(args.query) if args.query else DEFAULT_QUERIES

    try:
        entries = []
        for query in queries:
            entries.extend(
                search_openclipart(
                    query=query,
                    amount=args.per_query,
                    api_url=args.api_url,
                    fallback_api_url=None if args.no_fallback else FALLBACK_API_URL,
                )
            )
        result = export_openclipart_entries(
            entries=entries,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            image_size=args.image_size,
            clear_existing=args.clear_existing,
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"openclipart import failed: {exc}")
        return 1

    print(f"Wrote {result['image_count']} Openclipart image(s) to {args.output_dir}")
    print(f"Manifest: {args.manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Openclipart PNG images into Assets/ImageCorpus/Openclipart.")
    parser.add_argument("--query", action="append", default=[], help="Search term. Repeatable.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--per-query", type=int, default=DEFAULT_PER_QUERY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--no-fallback", action="store_true", help="Do not retry the HTTP API endpoint if HTTPS fails.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing PNGs in the output directory before writing the sampled corpus.",
    )
    return parser


def search_openclipart(
    query: str,
    amount: int,
    api_url: str = DEFAULT_API_URL,
    fallback_api_url: str | None = FALLBACK_API_URL,
) -> list[dict[str, Any]]:
    if amount <= 0:
        raise ValueError("amount must be positive")
    params = urllib.parse.urlencode({"query": query, "amount": amount})
    url = f"{api_url}?{params}"
    try:
        payload = _read_json_url(url)
    except (urllib.error.URLError, json.JSONDecodeError):
        if fallback_api_url is None:
            raise
        payload = _read_json_url(f"{fallback_api_url}?{params}")
    return collect_openclipart_entries(payload, query=query)


def collect_openclipart_entries(payload: Any, query: str) -> list[dict[str, Any]]:
    items = _payload_items(payload)
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        png_url = _find_png_url(item)
        if png_url is None or png_url in seen_urls:
            continue
        seen_urls.add(png_url)
        title = str(item.get("title") or item.get("name") or item.get("filename") or "openclipart")
        author = str(item.get("artist") or item.get("author") or item.get("user") or item.get("owner") or "unknown")
        detail_url = _find_detail_url(item)
        entries.append(
            {
                "query": query,
                "title": title,
                "title_slug": _slug(title),
                "author": author,
                "detail_url": detail_url,
                "png_url": png_url,
                "license": "CC0-1.0/Public Domain",
            }
        )
    return entries


def export_openclipart_entries(
    entries: Iterable[dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    limit: int,
    image_size: int | None = DEFAULT_IMAGE_SIZE,
    clear_existing: bool = False,
    downloader: Callable[[str], bytes] | None = None,
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

    downloader = downloader or _read_bytes_url
    images: list[dict[str, Any]] = []
    exported = 0
    seen_urls: set[str] = set()

    for entry in entries:
        png_url = str(entry.get("png_url") or "")
        if not png_url or png_url in seen_urls:
            continue
        seen_urls.add(png_url)
        image_bytes = downloader(png_url)
        file_name = f"openclipart_{exported + 1:06d}_{_slug(str(entry.get('title') or 'clipart'))}{IMAGE_SUFFIX}"
        output_path = output_dir / file_name
        width, height = _save_png(image_bytes=image_bytes, output_path=output_path, image_size=image_size)
        images.append(
            {
                "index": exported,
                "path": str(output_path),
                "file_name": file_name,
                "title": str(entry.get("title") or "openclipart"),
                "author": str(entry.get("author") or "unknown"),
                "query": str(entry.get("query") or ""),
                "detail_url": entry.get("detail_url"),
                "png_url": png_url,
                "license": "CC0-1.0/Public Domain",
                "width": width,
                "height": height,
            }
        )
        exported += 1
        if exported >= limit:
            break

    if exported == 0:
        raise ValueError("no Openclipart images found")

    manifest = {
        "version": 1,
        "source_dataset": "Openclipart",
        "source_reference": "https://openclipart.org/",
        "source_access": "Openclipart JSON search API",
        "license": "CC0-1.0/Public Domain",
        "license_note": "Openclipart states uploaded clipart uses Creative Commons Zero 1.0 Public Domain terms.",
        "output_dir": str(output_dir),
        "image_count": exported,
        "images": images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _payload_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("payload", "items", "results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _payload_items(value)
            if nested:
                return nested
    return []


def _find_png_url(item: dict[str, Any]) -> str | None:
    preferred_keys = (
        "png_full_lossy",
        "png_full",
        "png_2400px",
        "png_large",
        "png_medium",
        "png_thumb",
        "png",
        "download_png",
    )
    for key in preferred_keys:
        value = item.get(key)
        url = _url_from_value(value, suffix=".png")
        if url is not None:
            return url
    return _find_url_recursive(item, suffix=".png")


def _find_detail_url(item: dict[str, Any]) -> str | None:
    for key in ("detail_link", "detail_url", "url", "link", "permalink"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def _find_url_recursive(value: Any, suffix: str) -> str | None:
    if isinstance(value, str):
        return _url_from_value(value, suffix=suffix)
    if isinstance(value, dict):
        for nested_value in value.values():
            url = _find_url_recursive(nested_value, suffix=suffix)
            if url is not None:
                return url
    if isinstance(value, list):
        for nested_value in value:
            url = _find_url_recursive(nested_value, suffix=suffix)
            if url is not None:
                return url
    return None


def _url_from_value(value: Any, suffix: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith("//"):
        value = "https:" + value
    if not value.startswith(("http://", "https://")):
        return None
    parsed_path = urllib.parse.urlparse(value).path.lower()
    if parsed_path.endswith(suffix):
        return value
    if suffix == ".png" and "/image/" in parsed_path:
        return value
    return None


def _save_png(image_bytes: bytes, output_path: Path, image_size: int | None) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export Openclipart images") from exc

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        if image_size is not None:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((image_size, image_size), resampling)
            canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
            canvas.paste(image, offset)
            image = canvas
        image.save(output_path)
        return image.size


def _read_json_url(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if "json" not in content_type.lower():
        raise RuntimeError(
            f"Openclipart API returned {content_type or 'unknown content type'} instead of JSON for {url}. "
            "Use --api-url with an API-compatible mirror or place downloaded PNGs under Assets/ImageCorpus/Openclipart."
        )
    return json.loads(data.decode("utf-8"))


def _read_bytes_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _slug(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "openclipart"


if __name__ == "__main__":
    raise SystemExit(main())
