import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from Source.Synthetic.import_art_institute_chicago import (
    art_institute_entry_from_record,
    import_art_institute_images,
)


def _jpg_bytes(size=(12, 8), color=(40, 50, 60)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _record(record_id=1, title="The Bedroom", image_id="image-one"):
    return {
        "id": record_id,
        "title": title,
        "image_id": image_id,
        "is_public_domain": True,
        "artist_display": "Artist",
        "date_display": "1889",
        "medium_display": "Oil on canvas",
        "department_title": "Painting and Sculpture of Europe",
        "classification_title": "oil on canvas",
        "thumbnail": {"width": 1200, "height": 900},
    }


class ArtInstituteChicagoImportTest(unittest.TestCase):
    def test_extracts_public_domain_record(self):
        entry = art_institute_entry_from_record(_record(), iiif_url="https://www.artic.edu/iiif/2", image_size=512)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["title"], "The Bedroom")
        self.assertEqual(entry["image_id"], "image-one")
        self.assertIn("/full/!512,512/0/default.jpg", entry["image_url"])

    def test_import_writes_images_and_manifest(self):
        records = [_record(1, "The Bedroom", "image-one"), _record(2, "Still Life", "image-two")]

        def fake_searcher(page, rows, api_url, departments, classifications):
            if page > 1:
                return {"data": [], "pagination": {"total": len(records), "total_pages": 1}, "config": {"iiif_url": "https://www.artic.edu/iiif/2"}}
            return {
                "data": records,
                "pagination": {"total": len(records), "total_pages": 1},
                "config": {"iiif_url": "https://www.artic.edu/iiif/2"},
            }

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            manifest = import_art_institute_images(
                output_dir=temp_dir / "Assets" / "ImageCorpus" / "ArtInstituteChicago",
                manifest_path=temp_dir / "Outputs" / "ArtInstituteChicago" / "manifest.json",
                limit=2,
                rows=2,
                image_size=16,
                workers=2,
                searcher=fake_searcher,
                downloader=lambda url: _jpg_bytes(),
            )

            self.assertEqual(manifest["image_count"], 2)
            self.assertEqual(manifest["license"], "CC0-1.0")
            first_image = temp_dir / "Assets" / "ImageCorpus" / "ArtInstituteChicago" / "artic_000001_the_bedroom.jpg"
            self.assertTrue(first_image.exists())
            saved_manifest = json.loads((temp_dir / "Outputs" / "ArtInstituteChicago" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["images"][0]["width"], 12)
            self.assertEqual(saved_manifest["images"][0]["height"], 8)


if __name__ == "__main__":
    unittest.main()
