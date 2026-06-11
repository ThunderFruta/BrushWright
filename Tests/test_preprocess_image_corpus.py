from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Source.Synthetic.preprocess_image_corpus import preprocess_image_corpus, remove_uniform_border


class PreprocessImageCorpusTest(unittest.TestCase):
    def test_removes_uniform_border_and_normalizes_canvas(self) -> None:
        image = Image.new("RGB", (64, 64), (255, 255, 255))
        image.paste((20, 30, 40), (16, 18, 48, 46))

        result = remove_uniform_border(
            image,
            canvas_size=32,
            tolerance=10,
            padding=2,
            min_content_fraction=0.02,
        )

        self.assertTrue(result.cropped)
        self.assertEqual((16, 18, 48, 46), result.content_bbox)
        self.assertEqual((14, 16, 50, 48), result.crop_bbox)
        self.assertEqual((32, 32), result.image.size)

    def test_keeps_blank_image_uncropped(self) -> None:
        image = Image.new("RGB", (40, 24), (255, 255, 255))

        result = remove_uniform_border(image, canvas_size=32, tolerance=10, padding=2)

        self.assertFalse(result.cropped)
        self.assertEqual((0, 0, 40, 24), result.content_bbox)
        self.assertEqual((0, 0, 40, 24), result.crop_bbox)
        self.assertEqual((32, 32), result.image.size)

    def test_preprocess_writes_cropped_images_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            manifest_path = root / "manifest.json"
            input_dir.mkdir()

            image = Image.new("RGB", (64, 64), (250, 250, 250))
            image.paste((10, 40, 80), (20, 20, 44, 44))
            image.save(input_dir / "artwork.jpg")

            manifest = preprocess_image_corpus(
                input_dir=input_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                canvas_size=32,
                tolerance=8,
                padding=1,
                min_content_fraction=0.02,
            )

            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output_image = output_dir / "artwork_cropped.jpg"
            self.assertEqual(1, manifest["image_count"])
            self.assertEqual(1, saved_manifest["image_count"])
            self.assertTrue(output_image.exists())
            self.assertTrue(saved_manifest["images"][0]["cropped"])
            self.assertEqual(str(output_image), saved_manifest["images"][0]["path"])
            with Image.open(output_image) as processed:
                self.assertEqual((32, 32), processed.size)


if __name__ == "__main__":
    unittest.main()
