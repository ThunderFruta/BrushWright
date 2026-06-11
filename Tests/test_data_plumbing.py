import json
import tempfile
import unittest
from pathlib import Path

from Source.Synthetic.Templates import TEMPLATE_NAMES
from Source.Synthetic.build_sample import build_sample
from Source.Synthetic.generate_programs import generate_program, generate_program_bundle
from Source.Synthetic.split_strokes import split_program
from Source.Metrics.image_metrics import compare_images
from Source.Renderer.stroke_schema import load_stroke_program_json


class DataPlumbingTest(unittest.TestCase):
    def test_generate_program_is_deterministic(self):
        first = generate_program(seed=7, stroke_count=8)
        second = generate_program(seed=7, stroke_count=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first["strokes"]), 8)
        self.assertIn("template", first["metadata"])
        self.assertIn("style", first["metadata"])
        load_stroke_program_json(first)

    def test_each_template_is_deterministic_and_valid(self):
        for template_name in TEMPLATE_NAMES:
            with self.subTest(template=template_name):
                first = generate_program_bundle(
                    seed=11,
                    stroke_count=8,
                    base_count=5,
                    finishing_count=3,
                    template_name=template_name,
                    style_name="flat_vector",
                )
                second = generate_program_bundle(
                    seed=11,
                    stroke_count=8,
                    base_count=5,
                    finishing_count=3,
                    template_name=template_name,
                    style_name="flat_vector",
                )

                self.assertEqual(first, second)
                self.assertEqual(len(first["base_strokes"]["strokes"]), 5)
                self.assertEqual(len(first["finishing_strokes"]["strokes"]), 3)
                load_stroke_program_json(first["base_strokes"])
                load_stroke_program_json(first["finishing_strokes"])
                load_stroke_program_json(first["full_program"])

    def test_split_program_writes_expected_counts(self):
        program = generate_program(seed=8, stroke_count=10)
        split = split_program(program, base_count=6, finishing_count=4)

        self.assertEqual(len(split["base_strokes"]["strokes"]), 6)
        self.assertEqual(len(split["finishing_strokes"]["strokes"]), 4)
        self.assertEqual(split["manifest"]["withheld_start_index"], 6)
        load_stroke_program_json(split["base_strokes"])
        load_stroke_program_json(split["finishing_strokes"])

    def test_build_sample_creates_supervised_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir_name:
            output_dir = Path(output_dir_name)
            sample = build_sample(seed=9, output_dir=output_dir, stroke_count=8, base_count=5, finishing_count=3)

            for key in (
                "full_program",
                "base_strokes",
                "finishing_strokes",
                "draft_image",
                "finished_image",
                "draft_render_manifest",
                "finished_render_manifest",
                "split_manifest",
            ):
                self.assertTrue((output_dir / sample[key]).exists(), key)

            sample_json = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
            self.assertEqual(sample_json["base_count"], 5)
            self.assertEqual(sample_json["finishing_count"], 3)
            self.assertIn("template", sample_json)
            self.assertIn("style", sample_json)

            full_program = json.loads((output_dir / "full_program.json").read_text(encoding="utf-8"))
            base_program = json.loads((output_dir / "base_strokes.json").read_text(encoding="utf-8"))
            finishing_program = json.loads((output_dir / "finishing_strokes.json").read_text(encoding="utf-8"))
            self.assertEqual(full_program["strokes"], base_program["strokes"] + finishing_program["strokes"])

            metrics = compare_images(output_dir / sample["draft_image"], output_dir / sample["finished_image"])
            self.assertEqual(metrics["width"], 512)
            self.assertEqual(metrics["height"], 512)


if __name__ == "__main__":
    unittest.main()
