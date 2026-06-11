import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Source.Synthetic.import_artbench import export_artbench_rows, shuffled_subset


class ArtBenchImportTest(unittest.TestCase):
    def test_export_rows_writes_images_and_manifest(self):
        rows = [
            {"image": Image.new("RGB", (8, 10), (255, 0, 0)), "label": "post-impressionism"},
            {"image": Image.new("RGB", (12, 6), (0, 255, 0)), "label": "ukiyo e"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_dir = temp_dir / "Assets" / "ImageCorpus" / "ArtBench"
            manifest_path = temp_dir / "Outputs" / "ArtBench" / "manifest.json"

            manifest = export_artbench_rows(
                rows=rows,
                output_dir=output_dir,
                manifest_path=manifest_path,
                limit=2,
                styles=[],
                image_size=16,
            )

            self.assertEqual(manifest["image_count"], 2)
            self.assertTrue((output_dir / "artbench_000001_post_impressionism.png").exists())
            self.assertTrue((output_dir / "artbench_000002_ukiyo_e.png").exists())
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["style_counts"], {"post_impressionism": 1, "ukiyo_e": 1})
            self.assertEqual(saved_manifest["images"][0]["width"], 16)
            self.assertEqual(saved_manifest["images"][0]["height"], 16)

    def test_export_rows_filters_styles(self):
        rows = [
            {"image": Image.new("RGB", (8, 8)), "label": "expressionism"},
            {"image": Image.new("RGB", (8, 8)), "label": "baroque"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            manifest = export_artbench_rows(
                rows=rows,
                output_dir=temp_dir / "images",
                manifest_path=temp_dir / "manifest.json",
                limit=1,
                styles=["baroque"],
            )

            self.assertEqual(manifest["image_count"], 1)
            self.assertEqual(manifest["images"][0]["label_slug"], "baroque")

    def test_shuffled_subset_is_deterministic(self):
        paths = [Path(str(index)) for index in range(6)]

        self.assertEqual(shuffled_subset(paths, limit=3, seed=4), shuffled_subset(paths, limit=3, seed=4))


if __name__ == "__main__":
    unittest.main()
