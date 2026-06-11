from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from Source.Output.output_archive import prepare_latest_output_root


class OutputArchiveTest(unittest.TestCase):
    def test_existing_latest_output_moves_to_archive_before_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            latest = root / "Outputs" / "Latest" / "ImageDeltaStrokeCompilerV1"
            latest.mkdir(parents=True)
            (latest / "export_manifest.json").write_text("old", encoding="utf-8")

            prepared = prepare_latest_output_root(
                latest,
                now=datetime(2026, 6, 4, 12, 30, 0, tzinfo=timezone.utc),
            )

            archive = root / "Outputs" / "Archive" / "ImageDeltaStrokeCompilerV1" / "20260604T123000Z"
            self.assertEqual(prepared, latest)
            self.assertTrue(latest.exists())
            self.assertFalse((latest / "export_manifest.json").exists())
            self.assertEqual((archive / "export_manifest.json").read_text(encoding="utf-8"), "old")
            manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_latest_path"], str(latest))

    def test_other_latest_outputs_move_to_archive_before_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            active = root / "Outputs" / "Latest" / "GreedyStrokeOptimizerV1"
            sibling = root / "Outputs" / "Latest" / "ImageDeltaStrokeCompilerV1"
            other = root / "Outputs" / "Latest" / "TargetStrokeRetrievalV6Oracle"
            active.mkdir(parents=True)
            sibling.mkdir(parents=True)
            other.mkdir(parents=True)
            (active / "keep.txt").write_text("active-old", encoding="utf-8")
            (sibling / "export_manifest.json").write_text("sibling-old", encoding="utf-8")
            (other / "export_manifest.json").write_text("other-old", encoding="utf-8")

            prepare_latest_output_root(
                active,
                now=datetime(2026, 6, 4, 12, 30, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(active.exists())
            self.assertFalse((active / "keep.txt").exists())
            self.assertFalse(sibling.exists())
            self.assertFalse(other.exists())
            active_archive = root / "Outputs" / "Archive" / "GreedyStrokeOptimizerV1" / "20260604T123000Z"
            sibling_archive = root / "Outputs" / "Archive" / "ImageDeltaStrokeCompilerV1" / "20260604T123000Z"
            other_archive = root / "Outputs" / "Archive" / "TargetStrokeRetrievalV6Oracle" / "20260604T123000Z"
            self.assertEqual((active_archive / "keep.txt").read_text(encoding="utf-8"), "active-old")
            self.assertEqual((sibling_archive / "export_manifest.json").read_text(encoding="utf-8"), "sibling-old")
            self.assertEqual((other_archive / "export_manifest.json").read_text(encoding="utf-8"), "other-old")

    def test_custom_non_latest_output_is_not_archived(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            custom = root / "Outputs" / "ScratchRun"
            custom.mkdir(parents=True)
            (custom / "keep.txt").write_text("kept", encoding="utf-8")

            prepare_latest_output_root(custom)

            self.assertEqual((custom / "keep.txt").read_text(encoding="utf-8"), "kept")
            self.assertFalse((root / "Outputs" / "Archive").exists())

    def test_empty_latest_output_is_reused_without_archive_entry(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            latest = root / "Outputs" / "Latest" / "EmptyRun"
            latest.mkdir(parents=True)

            prepare_latest_output_root(latest)

            self.assertTrue(latest.exists())
            self.assertTrue((root / "Outputs" / "Archive").exists())
            self.assertFalse((root / "Outputs" / "Archive" / "EmptyRun").exists())


if __name__ == "__main__":
    unittest.main()
