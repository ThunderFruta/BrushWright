from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Source.Synthetic.import_google_quickdraw import (
    import_google_quickdraw,
    quickdraw_class_url,
    render_quickdraw_drawing,
)


class GoogleQuickDrawImportTest(unittest.TestCase):
    def test_class_url_encodes_spaces(self) -> None:
        self.assertEqual(
            "https://example.test/aircraft%20carrier.ndjson",
            quickdraw_class_url("aircraft carrier", "https://example.test"),
        )

    def test_render_quickdraw_drawing_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "drawing.png"
            render_quickdraw_drawing([[[0, 255], [0, 255]]], output_path, image_size=32, stroke_width=2)
            with Image.open(output_path) as image:
                self.assertEqual((32, 32), image.size)
                self.assertEqual("RGB", image.mode)

    def test_import_writes_per_class_images_and_manifest(self) -> None:
        records = [
            {"key_id": "a", "countrycode": "US", "recognized": True, "drawing": [[[0, 255], [0, 255]]]},
            {"key_id": "b", "countrycode": "CA", "recognized": False, "drawing": [[[255, 0], [0, 255]]]},
            {"key_id": "c", "countrycode": "GB", "recognized": True, "drawing": [[[10, 20], [30, 40]]]},
        ]

        import Source.Synthetic.import_google_quickdraw as quickdraw

        original_iter = quickdraw.iter_quickdraw_records
        quickdraw.iter_quickdraw_records = lambda _url: iter(records)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                manifest = import_google_quickdraw(
                    classes=["cat"],
                    per_class=2,
                    output_dir=root / "images",
                    manifest_path=root / "manifest.json",
                    image_size=32,
                    simplified_base_url="https://example.test",
                    recognized_only=True,
                )
                saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(2, manifest["image_count"])
                self.assertEqual(2, saved["image_count"])
                self.assertEqual("CC-BY-4.0", saved["license"])
                self.assertTrue((root / "images" / "quickdraw_cat_0001.png").exists())
                self.assertEqual("c", saved["images"][1]["key_id"])
        finally:
            quickdraw.iter_quickdraw_records = original_iter


if __name__ == "__main__":
    unittest.main()
