import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from Source.Synthetic.import_openclipart import collect_openclipart_entries, export_openclipart_entries


def _png_bytes(size=(8, 6), color=(10, 20, 30)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class OpenclipartImportTest(unittest.TestCase):
    def test_collect_entries_from_payload(self):
        payload = {
            "payload": [
                {
                    "title": "Simple Tree",
                    "artist": "openclipart",
                    "detail_link": "https://openclipart.org/detail/1/simple-tree",
                    "svg": {"png_full_lossy": "https://openclipart.org/image/800px/1.png"},
                },
                {
                    "title": "No PNG",
                    "svg": {"url": "https://openclipart.org/download/2/no-png.svg"},
                },
            ]
        }

        entries = collect_openclipart_entries(payload, query="tree")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Simple Tree")
        self.assertEqual(entries[0]["title_slug"], "simple_tree")
        self.assertEqual(entries[0]["license"], "CC0-1.0/Public Domain")

    def test_export_entries_writes_images_and_manifest(self):
        entries = [
            {"title": "Simple Tree", "author": "openclipart", "query": "tree", "png_url": "https://example.test/tree.png"},
            {"title": "Simple Flower", "author": "openclipart", "query": "flower", "png_url": "https://example.test/flower.png"},
        ]
        image_by_url = {
            "https://example.test/tree.png": _png_bytes(size=(8, 6)),
            "https://example.test/flower.png": _png_bytes(size=(5, 7)),
        }

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            manifest = export_openclipart_entries(
                entries=entries,
                output_dir=temp_dir / "Assets" / "ImageCorpus" / "Openclipart",
                manifest_path=temp_dir / "Outputs" / "Openclipart" / "manifest.json",
                limit=2,
                image_size=16,
                downloader=lambda url: image_by_url[url],
            )

            self.assertEqual(manifest["image_count"], 2)
            self.assertEqual(manifest["license"], "CC0-1.0/Public Domain")
            self.assertTrue((temp_dir / "Assets" / "ImageCorpus" / "Openclipart" / "openclipart_000001_simple_tree.png").exists())
            saved_manifest = json.loads((temp_dir / "Outputs" / "Openclipart" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["images"][0]["width"], 16)
            self.assertEqual(saved_manifest["images"][0]["height"], 16)


if __name__ == "__main__":
    unittest.main()
