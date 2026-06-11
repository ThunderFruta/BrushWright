import unittest

from Source.Renderer.stroke_schema import StrokeSchemaError, load_stroke_program_json


VALID_PROGRAM = {
    "version": 1,
    "canvas": {"width": 512, "height": 512},
    "metadata": {"sample_id": "test"},
    "strokes": [
        {
            "x": 0.5,
            "y": 0.5,
            "angle": 0.25,
            "length": 0.2,
            "width": 0.03,
            "color": [0.1, 0.2, 0.3],
            "opacity": 0.8,
            "brush": "flat_oil",
        }
    ],
}


class StrokeSchemaTest(unittest.TestCase):
    def test_valid_program_loads(self):
        program = load_stroke_program_json(VALID_PROGRAM)

        self.assertEqual(program.canvas.width, 512)
        self.assertEqual(program.canvas.height, 512)
        self.assertEqual(len(program.strokes), 1)
        self.assertEqual(program.strokes[0].brush, "flat_oil")

    def test_invalid_normalized_value_fails(self):
        data = dict(VALID_PROGRAM)
        data["strokes"] = [dict(VALID_PROGRAM["strokes"][0], x=1.25)]

        with self.assertRaisesRegex(StrokeSchemaError, "strokes\\[0\\].x"):
            load_stroke_program_json(data)

    def test_missing_required_field_fails(self):
        stroke = dict(VALID_PROGRAM["strokes"][0])
        del stroke["opacity"]
        data = dict(VALID_PROGRAM)
        data["strokes"] = [stroke]

        with self.assertRaisesRegex(StrokeSchemaError, "opacity is required"):
            load_stroke_program_json(data)


if __name__ == "__main__":
    unittest.main()

