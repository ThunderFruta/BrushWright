from __future__ import annotations

import unittest

from Source.PaintTransformerReference.synthesize_samples import _resolve_available_stroke_split


class PaintTransformerCountTest(unittest.TestCase):
    def test_keeps_requested_counts_when_enough_strokes_are_available(self) -> None:
        self.assertEqual((768, 256, False), _resolve_available_stroke_split(1200, 768, 256))

    def test_reduces_base_count_first_when_finishing_count_fits(self) -> None:
        self.assertEqual((735, 256, True), _resolve_available_stroke_split(991, 768, 256))

    def test_preserves_requested_ratio_when_finishing_count_does_not_fit(self) -> None:
        self.assertEqual((9, 3, True), _resolve_available_stroke_split(12, 768, 256))

    def test_requires_at_least_one_base_and_one_finishing_stroke(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_available_stroke_split(1, 768, 256)


if __name__ == "__main__":
    unittest.main()
