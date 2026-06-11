from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


def _stroke(x: float, y: float) -> dict:
    return {
        "x": x,
        "y": y,
        "angle": 0.0,
        "length": 0.02,
        "width": 0.01,
        "color": [1.0, 0.0, 0.0],
        "opacity": 1.0,
        "brush": "paint_transformer_rect",
    }


def _program(strokes: list[dict]) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {},
        "strokes": strokes,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_images(sample_dir: Path) -> None:
    from PIL import Image, ImageDraw

    draft = Image.new("RGB", (512, 512), (0, 0, 0))
    target = draft.copy()
    ImageDraw.Draw(target).rectangle((200, 200, 312, 312), fill=(255, 0, 0))
    draft.save(sample_dir / "draft.png")
    target.save(sample_dir / "target.png")
    target.save(sample_dir / "finished.png")


def _write_data_root(root: Path) -> None:
    split_root = root / "Train"
    sample_dir = split_root / "sample_a"
    sample_dir.mkdir(parents=True)
    _write_images(sample_dir)
    _write_json(sample_dir / "base_strokes.json", _program([]))
    _write_json(sample_dir / "finishing_strokes.json", _program([_stroke(0.5, 0.5)]))
    _write_json(sample_dir / "split_manifest.json", {"version": 1, "target_contract": "paint_transformer_original_image_target_v1"})
    _write_json(
        sample_dir / "sample.json",
        {
            "version": 1,
            "sample_id": "sample_a",
            "base_strokes": "base_strokes.json",
            "finishing_strokes": "finishing_strokes.json",
            "draft_image": "draft.png",
            "target_image": "target.png",
            "finished_image": "finished.png",
            "split_manifest": "split_manifest.json",
            "target_contract": "paint_transformer_original_image_target_v1",
        },
    )
    _write_json(
        split_root / "dataset_manifest.json",
        {
            "version": 1,
            "split": "Train",
            "target_contract": "paint_transformer_original_image_target_v1",
            "sample_count": 1,
            "samples": [{"sample_id": "sample_a", "path": "sample_a"}],
        },
    )


class VisualTeacherStrokesTest(unittest.TestCase):
    def test_generate_visual_teacher_strokes_updates_sample_metadata(self) -> None:
        from Source.Model.generate_visual_teacher_strokes import VisualTeacherConfig, generate_visual_teacher_strokes

        with tempfile.TemporaryDirectory() as root_name:
            data_root = Path(root_name) / "Data"
            _write_data_root(data_root)

            summary = generate_visual_teacher_strokes(
                VisualTeacherConfig(
                    data_root=data_root,
                    splits=("Train",),
                    max_strokes=2,
                    size_tiers=(80,),
                    detail_size_tiers=(32,),
                    angle_degrees=(0.0,),
                    opacities=(0.85,),
                    max_component_anchors=1,
                    max_point_anchors=1,
                    target_mad_threshold=0.0,
                )
            )

            sample_dir = data_root / "Train" / "sample_a"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            teacher = json.loads((sample_dir / "visual_teacher_strokes.json").read_text(encoding="utf-8"))
            manifest = json.loads((sample_dir / "visual_teacher_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(sample["visual_teacher_strokes"], "visual_teacher_strokes.json")
            self.assertEqual(sample["target_strokes_source"], "greedy_stroke_optimizer_v1")
            self.assertGreaterEqual(len(teacher["strokes"]), 1)
            self.assertGreater(manifest["estimated_mad_improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()
