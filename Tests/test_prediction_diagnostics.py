from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Source.Model.prediction_diagnostics import compute_prediction_diagnostics, stroke_statistics


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), color).save(path)


class PredictionDiagnosticsTest(unittest.TestCase):
    def test_marks_prediction_as_improved_when_closer_than_draft(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_image(root / "draft.png", (0, 0, 0))
            _write_image(root / "target.png", (255, 255, 255))
            _write_image(root / "predicted.png", (255, 255, 255))

            diagnostics = compute_prediction_diagnostics(
                draft_path=root / "draft.png",
                target_path=root / "target.png",
                predicted_path=root / "predicted.png",
                predicted_strokes=[_stroke()],
                target_strokes=[_stroke(x=0.0), _stroke(x=1.0)],
            )

            self.assertEqual(diagnostics["status"], "improved")
            self.assertTrue(diagnostics["visual_improved"])
            self.assertTrue(diagnostics["changed_enough"])
            self.assertIn("x", diagnostics["collapse_metrics"])
            self.assertLess(
                diagnostics["image_deltas"]["predicted_to_target"]["mean_absolute_difference"],
                diagnostics["image_deltas"]["draft_to_target"]["mean_absolute_difference"],
            )

    def test_marks_prediction_failed_when_not_closer_than_draft(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_image(root / "draft.png", (0, 0, 0))
            _write_image(root / "target.png", (255, 255, 255))
            _write_image(root / "predicted.png", (0, 0, 0))

            diagnostics = compute_prediction_diagnostics(
                draft_path=root / "draft.png",
                target_path=root / "target.png",
                predicted_path=root / "predicted.png",
                predicted_strokes=[_stroke()],
            )

            self.assertEqual(diagnostics["status"], "failed_low_pixel_change")
            self.assertFalse(diagnostics["visual_improved"])
            self.assertFalse(diagnostics["changed_enough"])

    def test_structure_gate_accepts_strong_masked_improvement_with_edge_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_image(root / "draft.png", (0, 0, 0))
            _write_image(root / "target.png", (255, 255, 255))
            _write_image(root / "predicted.png", (180, 180, 180))

            with patch(
                "Source.Model.prediction_diagnostics._structure_metrics",
                return_value={
                    "masked_mad_improvement": 0.12,
                    "gradient_improvement": -0.05,
                    "edge_overlap": 0.30,
                    "target_edge_count": 100.0,
                    "outside_mask_change": 0.0,
                },
            ):
                diagnostics = compute_prediction_diagnostics(
                    draft_path=root / "draft.png",
                    target_path=root / "target.png",
                    predicted_path=root / "predicted.png",
                    predicted_strokes=[_stroke()],
                )

        self.assertEqual(diagnostics["status"], "improved")
        self.assertTrue(diagnostics["structure_improved"])

    def test_structure_gate_rejects_weak_masked_improvement_with_gradient_loss(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_image(root / "draft.png", (0, 0, 0))
            _write_image(root / "target.png", (255, 255, 255))
            _write_image(root / "predicted.png", (180, 180, 180))

            with patch(
                "Source.Model.prediction_diagnostics._structure_metrics",
                return_value={
                    "masked_mad_improvement": 0.003,
                    "gradient_improvement": -0.05,
                    "edge_overlap": 0.30,
                    "target_edge_count": 100.0,
                    "outside_mask_change": 0.0,
                },
            ):
                diagnostics = compute_prediction_diagnostics(
                    draft_path=root / "draft.png",
                    target_path=root / "target.png",
                    predicted_path=root / "predicted.png",
                    predicted_strokes=[_stroke()],
                )

        self.assertEqual(diagnostics["status"], "failed_structure_noise")
        self.assertFalse(diagnostics["structure_improved"])

    def test_stroke_statistics_reports_brush_histogram_and_field_spread(self) -> None:
        stats = stroke_statistics([_stroke(brush="paint_transformer_rect"), _stroke(brush="flat_oil", x=0.75)])

        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["brush_histogram"], {"flat_oil": 1, "paint_transformer_rect": 1})
        self.assertEqual(stats["x"]["min"], 0.5)
        self.assertEqual(stats["x"]["max"], 0.75)
        self.assertGreater(stats["x"]["std"], 0.0)


def _stroke(brush: str = "paint_transformer_rect", x: float = 0.5) -> dict:
    return {
        "x": x,
        "y": 0.25,
        "angle": 0.5,
        "length": 0.1,
        "width": 0.02,
        "color": [0.2, 0.3, 0.4],
        "opacity": 1.0,
        "brush": brush,
    }


if __name__ == "__main__":
    unittest.main()
