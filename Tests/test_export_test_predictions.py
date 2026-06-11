from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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


def _program(strokes: list[dict]) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {},
        "strokes": strokes,
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (16, 16), color).save(path)


def _write_sample(split_root: Path, sample_id: str = "sample_a", missing_finished: bool = False) -> None:
    sample_dir = split_root / sample_id
    sample_dir.mkdir(parents=True)
    _write_json(
        sample_dir / "sample.json",
        {
            "version": 1,
            "sample_id": sample_id,
            "base_count": 4,
            "finishing_count": 4,
            "stroke_count_adjusted": False,
            "base_strokes": "base_strokes.json",
            "finishing_strokes": "finishing_strokes.json",
            "draft_image": "draft.png",
            "finished_image": "finished.png",
        },
    )
    _write_json(sample_dir / "base_strokes.json", _program([_stroke(index) for index in range(4)]))
    _write_json(sample_dir / "finishing_strokes.json", _program([_stroke(index) for index in range(4, 8)]))
    _write_image(sample_dir / "draft.png", (64, 96, 128))
    if not missing_finished:
        _write_image(sample_dir / "finished.png", (96, 128, 160))


def _write_manifest(split_root: Path) -> None:
    _write_json(
        split_root / "dataset_manifest.json",
        {
            "version": 1,
            "split": split_root.name,
            "sample_count": 1,
            "samples": [{"sample_id": "sample_a", "path": "sample_a"}],
        },
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for export tests")
class ExportTestPredictionsTest(unittest.TestCase):
    def test_target_guided_export_passes_finished_image_to_model(self) -> None:
        from Source.Model.export_test_predictions import ExportPredictionsConfig, export_test_predictions

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            data_root = root / "Data"
            split_root = data_root / "Test"
            split_root.mkdir(parents=True)
            _write_sample(split_root)
            _write_manifest(split_root)
            checkpoint = _write_checkpoint(root / "target_guided.pt", input_channels=9)

            exported = export_test_predictions(
                ExportPredictionsConfig(
                    data_root=data_root,
                    checkpoint=checkpoint,
                    output_root=root / "Outputs",
                    split="Test",
                    limit=1,
                    device="cpu",
                    render=False,
                )
            )

            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["status"], "not_rendered")
            self.assertTrue((root / "Outputs" / "Test" / "sample_a" / "predicted_finishing_strokes.json").exists())

    def test_target_guided_export_rejects_missing_target_image(self) -> None:
        from Source.Model.export_test_predictions import ExportPredictionsConfig, export_test_predictions

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            data_root = root / "Data"
            split_root = data_root / "Test"
            split_root.mkdir(parents=True)
            _write_sample(split_root, missing_finished=True)
            _write_manifest(split_root)
            checkpoint = _write_checkpoint(root / "target_guided.pt", input_channels=9)

            with self.assertRaisesRegex(ValueError, "requires target image"):
                export_test_predictions(
                    ExportPredictionsConfig(
                        data_root=data_root,
                        checkpoint=checkpoint,
                        output_root=root / "Outputs",
                        split="Test",
                        limit=1,
                        device="cpu",
                        render=False,
                    )
                )


def _write_checkpoint(path: Path, input_channels: int) -> Path:
    import torch

    from Source.Model import BrushWrightStrokePredictor, DraftImageEncoderConfig, StrokeEncoderConfig, StrokeTokenizer
    from Source.Model.stroke_decoder import StrokeChunkDecoderConfig

    encoder_config = StrokeEncoderConfig(model_dim=16, num_layers=1, num_heads=4, ff_dim=32, dropout=0.0, max_strokes=4)
    decoder_config = StrokeChunkDecoderConfig(model_dim=16, num_layers=1, num_heads=4, ff_dim=32, dropout=0.0, chunk_size=4, max_chunks=1)
    image_encoder_config = DraftImageEncoderConfig(
        model_dim=16,
        hidden_dim=16,
        grid_size=2,
        dropout=0.0,
        input_channels=input_channels,
    )
    model = BrushWrightStrokePredictor(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        image_encoder_config=image_encoder_config,
    )
    tokenizer = StrokeTokenizer(max_strokes=4, brush_vocab=encoder_config.brush_vocab)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_config": asdict(encoder_config),
            "decoder_config": asdict(decoder_config),
            "image_encoder_config": asdict(image_encoder_config),
            "tokenizer": {
                "brush_to_id": tokenizer.brush_to_id,
                "id_to_brush": {str(key): value for key, value in tokenizer.id_to_brush.items()},
            },
            "epoch": 0,
            "global_step": 0,
        },
        path,
    )
    return path


if __name__ == "__main__":
    unittest.main()
