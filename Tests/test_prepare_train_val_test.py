from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Source.Synthetic.prepare_train_val_test import prepare_splits


def _stroke(index: int) -> dict:
    value = (index % 100) / 100.0
    return {
        "x": value,
        "y": 0.25,
        "angle": 0.5,
        "length": 0.1,
        "width": 0.02,
        "color": [0.2, 0.3, 0.4],
        "opacity": 1.0,
        "brush": "paint_transformer_rect",
    }


def _program(stroke_count: int) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {"source": "test"},
        "strokes": [_stroke(index) for index in range(stroke_count)],
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_patch_image(
    path: Path,
    background: tuple[int, int, int],
    patch: tuple[int, int, int],
    box: tuple[int, int, int, int],
) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), background)
    ImageDraw.Draw(image).rectangle(box, fill=patch)
    image.save(path)


class PrepareTrainValTest(unittest.TestCase):
    def test_draft_image_completion_matches_base_stroke_split(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            render_dir = source_dir / "finished_render"
            frame_paths = []
            for index in range(4):
                frame_path = render_dir / "frames" / f"frame_{index:06d}.png"
                _write_image(frame_path, (index * 40, index * 40, index * 40))
                frame_paths.append(str(frame_path))
            _write_image(source_dir / "source.png", (12, 34, 56))
            _write_image(source_dir / "finished.png", (255, 255, 255))
            _write_json(source_dir / "full_program.json", _program(8))
            _write_json(
                source_dir / "finished_render_manifest.json",
                {
                    "version": 1,
                    "native_frame_count": 4,
                    "frames": frame_paths,
                },
            )
            _write_json(
                source_dir / "sample.json",
                {
                    "version": 1,
                    "sample_id": "paint_transformer_000001",
                    "source_image": "source.png",
                    "full_program": "full_program.json",
                    "finished_image": "finished.png",
                    "finished_render_manifest": "finished_render_manifest.json",
                },
            )

            prepare_splits(
                source_root=root / "Outputs",
                output_root=root / "Data",
                base_count=None,
                finishing_count=None,
                draft_image_completion_ratio=0.75,
                draft_image_min_completion=0.75,
                draft_image_max_completion=0.75,
                val_fraction=0.0,
                test_fraction=0.0,
            )

            sample_dir = root / "Data" / "Train" / "paint_transformer_000001"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            base_program = json.loads((sample_dir / "base_strokes.json").read_text(encoding="utf-8"))
            finishing_program = json.loads((sample_dir / "finishing_strokes.json").read_text(encoding="utf-8"))

            self.assertEqual(sample["base_count"], 6)
            self.assertEqual(sample["finishing_count"], 2)
            self.assertEqual(sample["completion_ratio"], 0.75)
            self.assertEqual(sample["draft_image_completion_ratio"], 0.75)
            self.assertEqual(sample["draft_stroke_completion_delta"], 0.0)
            self.assertEqual(len(base_program["strokes"]), 6)
            self.assertEqual(len(finishing_program["strokes"]), 2)

    def test_output_detail_pair_selects_native_frame_and_detail_split(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            draft_render_dir = source_dir / "draft_render"
            render_dir = source_dir / "finished_render"
            frame_paths = []
            for index in range(6):
                frame_path = render_dir / "frames" / f"frame_{index + 1:06d}.png"
                _write_image(frame_path, (index * 30, index * 30, index * 30))
                frame_paths.append(str(frame_path))
            _write_image(source_dir / "source.png", (12, 34, 56))
            _write_image(source_dir / "draft.png", (80, 80, 80))
            _write_image(source_dir / "finished.png", (255, 255, 255))
            detail_window_program = _program(8)
            detail_window_program["metadata"]["stroke_window"] = "detail"
            detail_window_program["metadata"]["selected_start_index"] = 10
            _write_json(source_dir / "full_program.json", detail_window_program)
            _write_json(source_dir / "base_strokes.json", _program(4))
            _write_json(source_dir / "finishing_strokes.json", _program(4))
            _write_json(
                draft_render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "renderer": "paint_transformer_native_inference",
                    "native_frame_index": 2,
                    "native_frame_count": 6,
                    "final_image": str(source_dir / "draft.png"),
                },
            )
            _write_json(
                render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "native_frame_count": 6,
                    "frames": frame_paths,
                },
            )
            _write_json(
                source_dir / "sample.json",
                {
                    "version": 1,
                    "sample_id": "paint_transformer_000001",
                    "source_image": "source.png",
                    "full_program": "full_program.json",
                    "base_strokes": "base_strokes.json",
                    "finishing_strokes": "finishing_strokes.json",
                    "draft_image": "draft.png",
                    "draft_render_manifest": "draft_render/render_manifest.json",
                    "finished_image": "finished.png",
                    "finished_render_manifest": "finished_render/render_manifest.json",
                },
            )

            prepare_splits(
                source_root=root / "Outputs",
                output_root=root / "Data",
                val_fraction=0.0,
                test_fraction=0.0,
                use_output_detail_pair=True,
            )

            sample_dir = root / "Data" / "Train" / "paint_transformer_000001"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            split_manifest = json.loads((sample_dir / "split_manifest.json").read_text(encoding="utf-8"))
            base_program = json.loads((sample_dir / "base_strokes.json").read_text(encoding="utf-8"))
            finishing_program = json.loads((sample_dir / "finishing_strokes.json").read_text(encoding="utf-8"))
            resplit_full = json.loads((sample_dir / "full_program.json").read_text(encoding="utf-8"))
            draft_manifest = json.loads((sample_dir / "draft_render" / "render_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(sample["target_contract"], "paint_transformer_original_image_target_v1")
            self.assertEqual(sample["draft_source"], "paint_transformer_native_frame")
            self.assertEqual(sample["target_source"], "source_original_image")
            self.assertEqual(sample["target_image"], "target.png")
            self.assertEqual(sample["target_selection_mode"], "visual_delta_v1")
            self.assertEqual(sample["target_selection_manifest"], "target_selection_manifest.json")
            self.assertEqual(sample["stroke_count"], 8)
            self.assertEqual(sample["base_count"], 5)
            self.assertEqual(sample["finishing_count"], 3)
            self.assertEqual(split_manifest["native_frame_index"], 3)
            self.assertEqual(draft_manifest["native_frame_index"], 3)
            self.assertEqual(base_program["metadata"]["stroke_window"], "detail")
            self.assertEqual(len(base_program["strokes"]), 5)
            self.assertEqual(len(finishing_program["strokes"]), 3)
            self.assertEqual(resplit_full["strokes"], base_program["strokes"] + finishing_program["strokes"])
            self.assertEqual((sample_dir / "draft.png").read_bytes(), Path(frame_paths[3]).read_bytes())
            self.assertEqual((sample_dir / "finished.png").read_bytes(), (source_dir / "finished.png").read_bytes())
            self.assertEqual((sample_dir / "target.png").read_bytes(), (source_dir / "source.png").read_bytes())

    def test_output_detail_pair_splits_hints_at_selected_draft_frame(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            draft_render_dir = source_dir / "draft_render"
            render_dir = source_dir / "finished_render"
            frame_paths = []
            for index in range(4):
                frame_path = render_dir / "frames" / f"frame_{index + 1:06d}.png"
                _write_image(frame_path, (index * 40, index * 40, index * 40))
                frame_paths.append(str(frame_path))
            _write_image(source_dir / "source.png", (12, 34, 56))
            _write_image(source_dir / "draft.png", (120, 120, 120))
            _write_image(source_dir / "finished.png", (255, 255, 255))
            _write_json(source_dir / "full_program.json", _program(8))
            _write_json(source_dir / "base_strokes.json", _program(5))
            _write_json(source_dir / "finishing_strokes.json", _program(3))
            _write_json(
                draft_render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "renderer": "paint_transformer_native_inference",
                    "native_frame_index": 2,
                    "native_frame_count": 4,
                    "final_image": str(source_dir / "draft.png"),
                },
            )
            _write_json(
                render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "native_frame_count": 4,
                    "frames": frame_paths,
                },
            )
            _write_json(
                source_dir / "sample.json",
                {
                    "version": 1,
                    "sample_id": "paint_transformer_000001",
                    "source_image": "source.png",
                    "base_count": 5,
                    "finishing_count": 3,
                    "full_program": "full_program.json",
                    "base_strokes": "base_strokes.json",
                    "finishing_strokes": "finishing_strokes.json",
                    "draft_image": "draft.png",
                    "draft_render_manifest": "draft_render/render_manifest.json",
                    "finished_image": "finished.png",
                    "finished_render_manifest": "finished_render/render_manifest.json",
                },
            )

            prepare_splits(
                source_root=root / "Outputs",
                output_root=root / "Data",
                draft_image_completion_ratio=0.75,
                draft_image_min_completion=0.75,
                draft_image_max_completion=0.75,
                val_fraction=0.0,
                test_fraction=0.0,
                use_output_detail_pair=True,
            )

            sample_dir = root / "Data" / "Train" / "paint_transformer_000001"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            base_program = json.loads((sample_dir / "base_strokes.json").read_text(encoding="utf-8"))
            finishing_program = json.loads((sample_dir / "finishing_strokes.json").read_text(encoding="utf-8"))

            self.assertEqual(sample["target_contract"], "paint_transformer_original_image_target_v1")
            self.assertEqual(sample["target_image"], "target.png")
            self.assertEqual(sample["base_count"], 6)
            self.assertEqual(sample["finishing_count"], 2)
            self.assertEqual(sample["completion_ratio"], 0.75)
            self.assertEqual(sample["draft_image_completion_ratio"], 0.75)
            self.assertEqual(sample["draft_stroke_completion_delta"], 0.0)
            self.assertEqual(len(base_program["strokes"]), 6)
            self.assertEqual(len(finishing_program["strokes"]), 2)

    def test_output_detail_pair_selects_target_hints_by_visual_delta(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            draft_render_dir = source_dir / "draft_render"
            render_dir = source_dir / "finished_render"
            frame_paths = []
            for index in range(4):
                frame_path = render_dir / "frames" / f"frame_{index + 1:06d}.png"
                _write_image(frame_path, (0, 0, 0))
                frame_paths.append(str(frame_path))
            _write_patch_image(source_dir / "source.png", (0, 0, 0), (255, 255, 255), (2, 2, 6, 6))
            _write_image(source_dir / "draft.png", (0, 0, 0))
            _write_patch_image(source_dir / "finished.png", (0, 0, 0), (255, 255, 255), (2, 2, 6, 6))
            available_program = _program(5)
            available_program["strokes"][0]["x"] = 0.25
            available_program["strokes"][0]["y"] = 0.25
            available_program["strokes"][0]["length"] = 0.08
            available_program["strokes"][0]["width"] = 0.04
            available_program["strokes"][1]["x"] = 0.75
            available_program["strokes"][1]["y"] = 0.75
            available_program["strokes"][2]["x"] = 0.80
            available_program["strokes"][2]["y"] = 0.75
            available_program["strokes"][3]["x"] = 0.85
            available_program["strokes"][3]["y"] = 0.75
            available_program["strokes"][4]["x"] = 0.90
            available_program["strokes"][4]["y"] = 0.75
            program = _program(4)
            program["strokes"] = available_program["strokes"][1:]
            _write_json(source_dir / "full_program.json", program)
            _write_json(source_dir / "available_strokes.json", available_program)
            _write_json(source_dir / "base_strokes.json", _program(2))
            _write_json(source_dir / "finishing_strokes.json", _program(2))
            _write_json(
                draft_render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "renderer": "paint_transformer_native_inference",
                    "native_frame_index": 1,
                    "native_frame_count": 4,
                    "final_image": str(source_dir / "draft.png"),
                },
            )
            _write_json(
                render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "native_frame_count": 4,
                    "frames": frame_paths,
                },
            )
            _write_json(
                source_dir / "sample.json",
                {
                    "version": 1,
                    "sample_id": "paint_transformer_000001",
                    "source_image": "source.png",
                    "full_program": "full_program.json",
                    "base_strokes": "base_strokes.json",
                    "finishing_strokes": "finishing_strokes.json",
                    "available_strokes": "available_strokes.json",
                    "draft_image": "draft.png",
                    "draft_render_manifest": "draft_render/render_manifest.json",
                    "finished_image": "finished.png",
                    "finished_render_manifest": "finished_render/render_manifest.json",
                },
            )

            prepare_splits(
                source_root=root / "Outputs",
                output_root=root / "Data",
                draft_image_completion_ratio=0.5,
                draft_image_min_completion=0.5,
                draft_image_max_completion=0.5,
                val_fraction=0.0,
                test_fraction=0.0,
                use_output_detail_pair=True,
            )

            sample_dir = root / "Data" / "Train" / "paint_transformer_000001"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            manifest = json.loads((sample_dir / "target_selection_manifest.json").read_text(encoding="utf-8"))
            finishing_program = json.loads((sample_dir / "finishing_strokes.json").read_text(encoding="utf-8"))

            self.assertEqual(sample["target_selection_mode"], "visual_delta_v1")
            self.assertIn(0, manifest["selected_source_indices"])
            self.assertIn(available_program["strokes"][0], finishing_program["strokes"])

    def test_output_detail_pair_resizes_original_target_for_visual_delta(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            draft_render_dir = source_dir / "draft_render"
            render_dir = source_dir / "finished_render"
            frame_paths = []
            for index in range(4):
                frame_path = render_dir / "frames" / f"frame_{index + 1:06d}.png"
                _write_image(frame_path, (index * 40, index * 40, index * 40), size=(16, 16))
                frame_paths.append(str(frame_path))
            _write_image(source_dir / "source.png", (12, 34, 56), size=(32, 24))
            _write_image(source_dir / "draft.png", (80, 80, 80), size=(16, 16))
            _write_image(source_dir / "finished.png", (255, 255, 255), size=(16, 16))
            _write_json(source_dir / "full_program.json", _program(8))
            _write_json(source_dir / "base_strokes.json", _program(4))
            _write_json(source_dir / "finishing_strokes.json", _program(4))
            _write_json(
                draft_render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "renderer": "paint_transformer_native_inference",
                    "native_frame_index": 2,
                    "native_frame_count": 4,
                    "final_image": str(source_dir / "draft.png"),
                },
            )
            _write_json(
                render_dir / "render_manifest.json",
                {
                    "version": 1,
                    "native_frame_count": 4,
                    "frames": frame_paths,
                },
            )
            _write_json(
                source_dir / "sample.json",
                {
                    "version": 1,
                    "sample_id": "paint_transformer_000001",
                    "source_image": "source.png",
                    "full_program": "full_program.json",
                    "base_strokes": "base_strokes.json",
                    "finishing_strokes": "finishing_strokes.json",
                    "draft_image": "draft.png",
                    "draft_render_manifest": "draft_render/render_manifest.json",
                    "finished_image": "finished.png",
                    "finished_render_manifest": "finished_render/render_manifest.json",
                },
            )

            prepare_splits(
                source_root=root / "Outputs",
                output_root=root / "Data",
                val_fraction=0.0,
                test_fraction=0.0,
                use_output_detail_pair=True,
            )

            sample_dir = root / "Data" / "Train" / "paint_transformer_000001"
            sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))

            self.assertEqual(sample["target_contract"], "paint_transformer_original_image_target_v1")
            self.assertEqual(sample["target_source"], "source_original_image")
            self.assertEqual(sample["target_image"], "target.png")
            from PIL import Image

            with Image.open(sample_dir / "target.png") as target:
                self.assertEqual(target.size, (16, 16))

    def test_detail_window_render_context_uses_available_underpainting(self) -> None:
        from Source.Synthetic.prepare_train_val_test import _render_context_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            source_dir = root / "Outputs" / "paint_transformer_000001"
            source_dir.mkdir(parents=True)
            available_program = _program(12)
            selected_strokes = available_program["strokes"][8:12]
            full_program = _program(4)
            full_program["metadata"]["stroke_window"] = "detail"
            full_program["metadata"]["selected_start_index"] = 8
            _write_json(source_dir / "available_strokes.json", available_program)

            render_strokes, render_base_count, render_context_count = _render_context_strokes(
                source_dir=source_dir,
                source_sample={"available_strokes": "available_strokes.json"},
                full_program=full_program,
                selected_strokes=selected_strokes,
                base_count=3,
            )

            self.assertEqual(render_context_count, 8)
            self.assertEqual(render_base_count, 11)
            self.assertEqual(render_strokes, available_program["strokes"][:12])



if __name__ == "__main__":
    unittest.main()
