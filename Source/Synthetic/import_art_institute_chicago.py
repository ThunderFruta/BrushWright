"""Import Art Institute of Chicago Open Access images into BrushWright.

The Art Institute of Chicago offers public-domain collection images under CC0.
This importer samples public-domain records with image IDs from the public API,
downloads resized IIIF JPEGs, and records a reproducibility manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_API_URL = "https://api.artic.edu/api/v1/artworks"
DEFAULT_LIMIT = 5_000
DEFAULT_ROWS = 100
DEFAULT_OUTPUT_DIR = Path("Assets/ImageCorpus/ArtInstituteChicago")
DEFAULT_MANIFEST_PATH = Path("Outputs/ArtInstituteChicago/art_institute_chicago_import_manifest.json")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DELAY_SECONDS = 0.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKERS = 16
IMAGE_SUFFIX = ".jpg"
USER_AGENT = "BrushWright/0.1"
FIELDS = (
    "id,title,image_id,artist_display,date_display,medium_display,"
    "department_title,classification_title,is_public_domain,thumbnail"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    socket.setdefaulttimeout(args.timeout_seconds)

    try:
        result = import_art_institute_images(
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            rows=args.rows,
            page=args.page,
            image_size=args.image_size,
            delay_seconds=args.delay_seconds,
            clear_existing=args.clear_existing,
            api_url=args.api_url,
            workers=args.workers,
            department=args.department,
            classification=args.classification,
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"art institute import failed: {exc}")
        return 1

    print(f"Wrote {result['image_count']} Art Institute of Chicago image(s) to {args.output_dir}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Art Institute of Chicago CC0 images.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--department", action="append", default=[], help="Optional department filter. Repeatable.")
    parser.add_argument("--classification", action="append", default=[], help="Optional classification filter. Repeatable.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing imported JPGs before writing the sampled corpus.",
    )
    return parser


def import_art_institute_images(
    output_dir: Path,
    manifest_path: Path,
    limit: int = DEFAULT_LIMIT,
    rows: int = DEFAULT_ROWS,
    page: int = 1,
    image_size: int | None = DEFAULT_IMAGE_SIZE,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    clear_existing: bool = False,
    api_url: str = DEFAULT_API_URL,
    workers: int = DEFAULT_WORKERS,
    department: Sequence[str] = (),
    classification: Sequence[str] = (),
    searcher: Callable[[int, int, str, Sequence[str], Sequence[str]], dict[str, Any]] | None = None,
    downloader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if page <= 0:
        raise ValueError("page must be positive")
    if image_size is not None and image_size <= 0:
        raise ValueError("image_size must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    if workers <= 0:
        raise ValueError("workers must be positive")

    output_dir = output_dir.expanduser()
    manifest_path = manifest_path.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for existing in output_dir.glob(f"*{IMAGE_SUFFIX}"):
            existing.unlink()

    searcher = searcher or search_art_institute
    downloader = downloader or _read_bytes_url

    images: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    current_page = page
    total_pages: int | None = None
    total_records: int | None = None
    scanned = 0
    exported = 0
    iiif_url = "https://www.artic.edu/iiif/2"

    while exported < limit:
        try:
            payload = searcher(current_page, rows, api_url, department, classification)
        except urllib.error.HTTPError as exc:
            if exported > 0:
                failed.append({"id": "api", "title": "Art Institute API", "error": f"HTTP {exc.code}: {exc.reason}"})
                print(f"stopping early after {exported}/{limit} images: HTTP {exc.code} {exc.reason}", flush=True)
                break
            raise

        config = payload.get("config", {})
        iiif_url = str(config.get("iiif_url") or iiif_url).rstrip("/")
        pagination = payload.get("pagination", {})
        total_pages = _int_or_none(pagination.get("total_pages")) or total_pages
        total_records = _int_or_none(pagination.get("total")) or total_records
        records = payload.get("data", [])
        if not records:
            break

        department_filters = {_slug(value) for value in department}
        classification_filters = {_slug(value) for value in classification}
        jobs: list[dict[str, Any]] = []
        for record in records:
            scanned += 1
            entry = art_institute_entry_from_record(record, iiif_url=iiif_url, image_size=image_size)
            if entry is None:
                continue
            if department_filters and _slug(entry["department_title"]) not in department_filters:
                continue
            if classification_filters and _slug(entry["classification_title"]) not in classification_filters:
                continue
            record_id = str(entry["id"])
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            job_index = exported + len(jobs)
            if job_index >= limit:
                break
            file_name = f"artic_{job_index + 1:06d}_{_slug(entry['title'])}{IMAGE_SUFFIX}"
            jobs.append({"index": job_index, "entry": entry, "file_name": file_name, "output_path": output_dir / file_name})

        for result in _download_jobs(jobs, downloader, workers):
            if result.get("error"):
                failed.append({"id": result["id"], "title": result["title"], "error": result["error"]})
                continue
            images.append(result["image"])
            exported += 1
            if exported % 100 == 0:
                print(f"downloaded {exported}/{limit} Art Institute images", flush=True)
            if exported >= limit:
                break

        if total_pages is not None and current_page >= total_pages:
            break
        current_page += 1
        if delay_seconds:
            time.sleep(delay_seconds)

    if exported == 0:
        raise ValueError("no Art Institute of Chicago images were downloaded")

    manifest = {
        "version": 1,
        "source_dataset": "Art Institute of Chicago Open Access",
        "source_reference": "https://www.artic.edu/open-access",
        "source_api": api_url,
        "query": "is_public_domain=true AND image_id exists",
        "department_filters": list(department),
        "classification_filters": list(classification),
        "page": page,
        "rows": rows,
        "limit": limit,
        "scanned_rows": scanned,
        "total_records": total_records,
        "total_pages": total_pages,
        "output_dir": str(output_dir),
        "image_count": exported,
        "failed_count": len(failed),
        "license": "CC0-1.0",
        "license_note": "Art Institute of Chicago Open Access public-domain images and data are offered under CC0; website terms may still apply.",
        "images": images,
        "failed": failed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def search_art_institute(
    page: int,
    rows: int,
    api_url: str = DEFAULT_API_URL,
    departments: Sequence[str] = (),
    classifications: Sequence[str] = (),
) -> dict[str, Any]:
    params: list[tuple[str, str | int]] = [
        ("limit", rows),
        ("page", page),
        ("fields", FIELDS),
    ]
    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def art_institute_entry_from_record(record: dict[str, Any], iiif_url: str, image_size: int | None) -> dict[str, Any] | None:
    if not record.get("is_public_domain"):
        return None
    image_id = record.get("image_id")
    if not isinstance(image_id, str) or not image_id.strip():
        return None
    title = str(record.get("title") or "artic")
    thumbnail = record.get("thumbnail") if isinstance(record.get("thumbnail"), dict) else {}
    width = _int_or_none(thumbnail.get("width"))
    height = _int_or_none(thumbnail.get("height"))
    image_url = _iiif_image_url(iiif_url, image_id=image_id, image_size=image_size)
    return {
        "id": record.get("id"),
        "title": title,
        "image_id": image_id,
        "image_url": image_url,
        "artist_display": str(record.get("artist_display") or ""),
        "date_display": str(record.get("date_display") or ""),
        "medium_display": str(record.get("medium_display") or ""),
        "department_title": str(record.get("department_title") or ""),
        "classification_title": str(record.get("classification_title") or ""),
        "source_width": width,
        "source_height": height,
    }


def _download_jobs(jobs: Sequence[dict[str, Any]], downloader: Callable[[str], bytes], workers: int) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers == 1 or len(jobs) == 1:
        return [_download_job(job, downloader) for job in jobs]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(_download_job, job, downloader): int(job["index"]) for job in jobs}
        for future in concurrent.futures.as_completed(future_to_index):
            results.append(future.result())
    return sorted(results, key=lambda result: int(result.get("index", 0)))


def _download_job(job: dict[str, Any], downloader: Callable[[str], bytes]) -> dict[str, Any]:
    entry = job["entry"]
    output_path = Path(job["output_path"])
    try:
        if output_path.exists() and output_path.stat().st_size > 0:
            width, height = _image_size(output_path)
        else:
            image_bytes = downloader(entry["image_url"])
            width, height = _save_jpeg(image_bytes, output_path)
    except (OSError, RuntimeError, urllib.error.URLError, ValueError) as exc:
        return {"index": job["index"], "id": str(entry["id"]), "title": entry["title"], "error": str(exc)}
    return {
        "index": job["index"],
        "id": str(entry["id"]),
        "title": entry["title"],
        "error": "",
        "image": {
            "index": job["index"],
            "path": str(output_path),
            "file_name": job["file_name"],
            "id": entry["id"],
            "title": entry["title"],
            "image_id": entry["image_id"],
            "image_url": entry["image_url"],
            "artist_display": entry["artist_display"],
            "date_display": entry["date_display"],
            "medium_display": entry["medium_display"],
            "department_title": entry["department_title"],
            "classification_title": entry["classification_title"],
            "license": "CC0-1.0",
            "width": width,
            "height": height,
            "source_width": entry["source_width"],
            "source_height": entry["source_height"],
        },
    }


def _iiif_image_url(iiif_url: str, image_id: str, image_size: int | None) -> str:
    if image_size is None:
        size = "full"
    else:
        size = f"!{image_size},{image_size}"
    return f"{iiif_url.rstrip('/')}/{image_id}/full/{size}/0/default.jpg"


def _save_jpeg(image_bytes: bytes, output_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export Art Institute images") from exc
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=92, optimize=True)
        return image.size


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to inspect Art Institute images") from exc
    with Image.open(path) as image:
        return image.size


def _read_bytes_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _slug(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "artic"


if __name__ == "__main__":
    raise SystemExit(main())
