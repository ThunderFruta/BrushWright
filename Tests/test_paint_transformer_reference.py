import unittest

from Source.PaintTransformerReference.strokes import (
    PAINT_TRANSFORMER_BRUSH,
    paint_transformer_param_to_stroke,
)
from Source.Renderer.stroke_schema import load_stroke_program_json


class PaintTransformerReferenceTest(unittest.TestCase):
    def test_param_converts_to_valid_brushwright_stroke(self):
        stroke = paint_transformer_param_to_stroke(
            param=[0.5, 0.5, 0.25, 0.1, 0.5, 1.2, -0.1, 0.4],
            patch_x=1,
            patch_y=2,
            patch_count=4,
        )

        self.assertEqual(stroke["brush"], PAINT_TRANSFORMER_BRUSH)
        self.assertEqual(stroke["color"], [1.0, 0.0, 0.4])
        self.assertGreater(stroke["length"], stroke["width"])
        load_stroke_program_json(
            {
                "version": 1,
                "canvas": {"width": 512, "height": 512},
                "strokes": [stroke],
                "metadata": {},
            }
        )

    def test_vertical_stroke_rotates_major_axis(self):
        stroke = paint_transformer_param_to_stroke(
            param=[0.5, 0.5, 0.1, 0.25, 0.0, 0.2, 0.3, 0.4],
            patch_x=0,
            patch_y=0,
            patch_count=1,
        )

        self.assertEqual(stroke["angle"], 0.25)
        self.assertEqual(stroke["length"], 0.25)
        self.assertEqual(stroke["width"], 0.1)


if __name__ == "__main__":
    unittest.main()
