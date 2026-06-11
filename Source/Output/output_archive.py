"""Latest/archive management for generated BrushWright outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from pathlib import Path


OUTPUTS_DIR_NAME = "Outputs"
LATEST_DIR_NAME = "Latest"
ARCHIVE_DIR_NAME = "Archive"


def prepare_latest_output_root(output_root: Path, *, now: datetime | None = None) -> Path:
    """Archive populated Outputs/Latest run directories before a new run writes."""

    output_root = Path(output_root).expanduser()
    layout = _latest_layout(output_root)
    if layout is None:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    outputs_root, relative_name = layout
    archive_root = outputs_root / ARCHIVE_DIR_NAME
    latest_root = outputs_root / LATEST_DIR_NAME
    archived_at = now or datetime.now(timezone.utc)
    _archive_other_latest_roots(
        latest_root=latest_root,
        archive_root=archive_root,
        current_relative_name=relative_name,
        archived_at=archived_at,
    )
    if output_root.exists() and _has_contents(output_root):
        _archive_path(output_root, archive_root / relative_name, archived_at=archived_at)
    output_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _latest_layout(output_root: Path) -> tuple[Path, Path] | None:
    parts = output_root.parts
    for index in range(len(parts) - 1):
        if parts[index] == OUTPUTS_DIR_NAME and parts[index + 1] == LATEST_DIR_NAME:
            relative_parts = parts[index + 2 :]
            if not relative_parts:
                return None
            outputs_root = Path(*parts[: index + 1])
            return outputs_root, Path(*relative_parts)
    return None


def _archive_destination(archive_root: Path, relative_name: Path, archived_at: datetime) -> Path:
    timestamp = archived_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = archive_root / relative_name / timestamp
    if not destination.exists():
        return destination
    suffix = 2
    while True:
        candidate = archive_root / relative_name / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _archive_other_latest_roots(
    *,
    latest_root: Path,
    archive_root: Path,
    current_relative_name: Path,
    archived_at: datetime,
) -> None:
    if not latest_root.exists():
        return
    current_top_name = current_relative_name.parts[0]
    for latest_child in sorted(latest_root.iterdir(), key=lambda path: path.name):
        if latest_child.name == current_top_name:
            continue
        if not _has_contents(latest_child):
            continue
        _archive_path(latest_child, archive_root / latest_child.name, archived_at=archived_at)


def _archive_path(path: Path, archive_root: Path, *, archived_at: datetime) -> None:
    archive_destination = _archive_destination(archive_root.parent, archive_root.name, archived_at)
    archive_destination.parent.mkdir(parents=True, exist_ok=True)
    source_latest_path = path
    if path.is_dir():
        shutil.move(str(path), str(archive_destination))
    else:
        archive_destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive_destination / path.name))
    _write_archive_manifest(
        archive_destination / "archive_manifest.json",
        source_latest_path=source_latest_path,
        archived_at=archived_at,
    )


def _has_contents(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def _write_archive_manifest(path: Path, *, source_latest_path: Path, archived_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_latest_path": str(source_latest_path),
                "archived_at": archived_at.astimezone(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
