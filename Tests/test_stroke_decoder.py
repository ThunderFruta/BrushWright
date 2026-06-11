from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _program(strokes: list[dict]) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {},
        "strokes": strokes,
    }


def _stroke(index: int, brush: str = "paint_transformer_rect") -> dict:
    value = (index % 100) / 100.0
    return {
        "x": value,
        "y": 0.25,
        "angle": 0.5,
        "length": 0.1,
        "width": 0.02,
        "color": [0.2, 0.3, 0.4],
        "opacity": 1.0,
        "brush": brush,
    }


def _write_sample(split_root: Path, sample_id: str, base_count: int, finishing_count: int, adjusted: bool) -> None:
    sample_dir = split_root / sample_id
    sample_dir.mkdir(parents=True)
    sample = {
        "version": 1,
        "sample_id": sample_id,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "stroke_count_adjusted": adjusted,
        "base_strokes": "base_strokes.json",
        "finishing_strokes": "finishing_strokes.json",
        "draft_image": "draft.png",
        "finished_image": "finished.png",
    }
    _write_json(sample_dir / "sample.json", sample)
    _write_json(sample_dir / "base_strokes.json", _program([_stroke(index) for index in range(base_count)]))
    _write_json(sample_dir / "finishing_strokes.json", _program([_stroke(index) for index in range(finishing_count)]))
    _write_image(sample_dir / "draft.png", (64, 96, 128))
    _write_image(sample_dir / "finished.png", (96, 128, 160))


def _write_manifest(split_root: Path, samples: list[dict]) -> None:
    _write_json(
        split_root / "dataset_manifest.json",
        {
            "version": 1,
            "split": split_root.name,
            "sample_count": len(samples),
            "samples": samples,
        },
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (16, 16), color).save(path)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model decoder tests")
class StrokeDecoderTest(unittest.TestCase):
    def test_decoder_output_shapes_and_ranges(self) -> None:
        import torch

        from Source.Model import (
            StrokeChunkDecoder,
            StrokeChunkDecoderConfig,
            StrokeEncoder,
            StrokeEncoderConfig,
            StrokeTokenizer,
        )

        tokenizer = StrokeTokenizer(max_strokes=4)
        tokens = tokenizer.encode_program(_program([_stroke(1), _stroke(2)]))
        encoder = StrokeEncoder(StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4))
        decoder = StrokeChunkDecoder(
            StrokeChunkDecoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, chunk_size=8, max_chunks=16)
        )

        encoder_output = encoder(tokens.numeric, tokens.brush_ids, tokens.padding_mask)
        output = decoder(encoder_output, torch.tensor([0]))

        self.assertEqual(output.pred_numeric.shape, (1, 8, 9))
        self.assertEqual(output.pred_brush_logits.shape, (1, 8, tokenizer.vocab_size))
        self.assertGreaterEqual(output.pred_numeric.min().item(), 0.0)
        self.assertLessEqual(output.pred_numeric.max().item(), 1.0)
        self.assertGreaterEqual(output.pred_numeric[..., 3].min().item(), decoder.config.min_length)
        self.assertLessEqual(output.pred_numeric[..., 3].max().item(), decoder.config.max_length)
        self.assertGreaterEqual(output.pred_numeric[..., 4].min().item(), decoder.config.min_width)
        self.assertLessEqual(output.pred_numeric[..., 4].max().item(), decoder.config.max_width)

    def test_decoder_accepts_padded_base_strokes(self) -> None:
        import torch

        from Source.Model import StrokeChunkDecoder, StrokeChunkDecoderConfig, StrokeEncoder, StrokeEncoderConfig, StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=4)
        tokens = tokenizer.encode_programs(
            [
                _program([_stroke(1)]),
                _program([_stroke(2), _stroke(3), _stroke(4)]),
            ]
        )
        encoder = StrokeEncoder(StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4))
        decoder = StrokeChunkDecoder(
            StrokeChunkDecoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, chunk_size=8, max_chunks=16)
        )

        encoder_output = encoder(tokens.numeric, tokens.brush_ids, tokens.padding_mask)
        output = decoder(encoder_output, torch.tensor([0, 64]))

        self.assertEqual(output.pred_numeric.shape, (2, 8, 9))
        self.assertEqual(output.pred_brush_logits.shape, (2, 8, tokenizer.vocab_size))

    def test_spatial_decoder_uses_anchored_xy_and_constrained_sizes(self) -> None:
        import torch

        from Source.Model import StrokeChunkDecoder, StrokeChunkDecoderConfig, StrokeEncoderOutput

        config = StrokeChunkDecoderConfig(
            model_dim=32,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
            chunk_size=8,
            max_chunks=16,
            query_mode="spatial",
            spatial_grid_size=2,
        )
        decoder = StrokeChunkDecoder(config)
        encoder_output = StrokeEncoderOutput(
            features=torch.randn(1, 6, 32),
            pooled=torch.zeros(1, 32),
            padding_mask=torch.zeros(1, 6, dtype=torch.bool),
        )

        output = decoder(encoder_output, torch.tensor([0]))

        self.assertEqual(output.pred_numeric.shape, (1, 8, 9))
        self.assertTrue(torch.all(output.pred_numeric[..., 0:2] >= 0.0))
        self.assertTrue(torch.all(output.pred_numeric[..., 0:2] <= 1.0))
        self.assertGreaterEqual(output.pred_numeric[..., 3].min().item(), config.min_length)
        self.assertLessEqual(output.pred_numeric[..., 3].max().item(), config.max_length)
        self.assertGreaterEqual(output.pred_numeric[..., 4].min().item(), config.min_width)
        self.assertLessEqual(output.pred_numeric[..., 4].max().item(), config.max_width)

    def test_combined_predictor_accepts_stroke_batch(self) -> None:
        import torch

        from Source.Model import (
            BrushWrightStrokePredictor,
            DraftImageEncoderConfig,
            StrokeBatch,
            StrokeEncoderConfig,
            StrokeTokenBatch,
            collate_stroke_chunks,
        )
        from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=64, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_a", "path": "sample_a"}])
            dataset = BrushWrightStrokeDataset(split_root, chunk_size=8, max_base_strokes=4)
            batch = collate_stroke_chunks([dataset[0]])

            self.assertIsInstance(batch, StrokeBatch)
            self.assertIsInstance(batch.base_tokens, StrokeTokenBatch)
            model = BrushWrightStrokePredictor(
                encoder_config=StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4),
                decoder_config=StrokeChunkDecoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, chunk_size=8, max_chunks=16),
                image_encoder_config=DraftImageEncoderConfig(model_dim=32, hidden_dim=16, grid_size=2, dropout=0.0, input_channels=3),
            )
            output = model(batch)

            self.assertEqual(output.pred_numeric.shape, (1, 8, 9))
            self.assertEqual(output.pred_brush_logits.shape[-1], 8)
            self.assertTrue(torch.all(output.pred_numeric >= 0.0))
            self.assertTrue(torch.all(output.pred_numeric <= 1.0))

            changed_goal_batch = StrokeBatch(
                base_tokens=batch.base_tokens,
                target_numeric=batch.target_numeric,
                target_brush_ids=batch.target_brush_ids,
                target_padding_mask=batch.target_padding_mask,
                sample_ids=batch.sample_ids,
                chunk_starts=batch.chunk_starts,
                chunk_ends=batch.chunk_ends,
                stroke_count_adjusted=batch.stroke_count_adjusted,
                draft_images=batch.draft_images,
                goal_images=1.0 - batch.goal_images,
                error_maps=1.0 - batch.error_maps,
            )
            changed_output = model(changed_goal_batch)
            self.assertTrue(torch.equal(output.pred_numeric, changed_output.pred_numeric))
            self.assertTrue(torch.equal(output.pred_brush_logits, changed_output.pred_brush_logits))

    def test_target_guided_predictor_changes_when_goal_image_changes(self) -> None:
        import torch

        from Source.Model import (
            BrushWrightStrokePredictor,
            DraftImageEncoderConfig,
            StrokeBatch,
            StrokeEncoderConfig,
            collate_stroke_chunks,
        )
        from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
        from Source.Model import BrushWrightStrokeDataset

        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=64, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_a", "path": "sample_a"}])
            dataset = BrushWrightStrokeDataset(split_root, chunk_size=8, max_base_strokes=4)
            batch = collate_stroke_chunks([dataset[0]])
            model = BrushWrightStrokePredictor(
                encoder_config=StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4),
                decoder_config=StrokeChunkDecoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, chunk_size=8, max_chunks=16),
                image_encoder_config=DraftImageEncoderConfig(model_dim=32, hidden_dim=16, grid_size=2, dropout=0.0, input_channels=9),
            )
            model.eval()

            changed_goal_batch = StrokeBatch(
                base_tokens=batch.base_tokens,
                target_numeric=batch.target_numeric,
                target_brush_ids=batch.target_brush_ids,
                target_padding_mask=batch.target_padding_mask,
                sample_ids=batch.sample_ids,
                chunk_starts=batch.chunk_starts,
                chunk_ends=batch.chunk_ends,
                stroke_count_adjusted=batch.stroke_count_adjusted,
                draft_images=batch.draft_images,
                goal_images=1.0 - batch.goal_images,
                error_maps=torch.abs((1.0 - batch.goal_images) - batch.draft_images),
            )

            with torch.no_grad():
                output = model(batch)
                changed_output = model(changed_goal_batch)

            self.assertFalse(torch.equal(output.pred_numeric, changed_output.pred_numeric))
            self.assertFalse(torch.equal(output.pred_brush_logits, changed_output.pred_brush_logits))

    def test_predictor_is_deterministic_in_eval_mode(self) -> None:
        import torch

        from Source.Model import (
            BrushWrightStrokePredictor,
            DraftImageEncoderConfig,
            StrokeEncoderConfig,
            StrokeTokenizer,
            collate_stroke_chunks,
        )
        from Source.Model.stroke_decoder import StrokeChunkDecoderConfig
        from Source.Model import BrushWrightStrokeDataset

        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=64, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_a", "path": "sample_a"}])
            dataset = BrushWrightStrokeDataset(
                split_root,
                chunk_size=8,
                max_base_strokes=4,
                tokenizer=StrokeTokenizer(max_strokes=4),
            )
            batch = collate_stroke_chunks([dataset[0]])
            model = BrushWrightStrokePredictor(
                encoder_config=StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.1, max_strokes=4),
                decoder_config=StrokeChunkDecoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.1, chunk_size=8, max_chunks=16),
                image_encoder_config=DraftImageEncoderConfig(model_dim=32, hidden_dim=16, grid_size=2, dropout=0.1, input_channels=3),
            )
            model.eval()

            with torch.no_grad():
                first = model(batch)
                second = model(batch)

            self.assertTrue(torch.equal(first.pred_numeric, second.pred_numeric))
            self.assertTrue(torch.equal(first.pred_brush_logits, second.pred_brush_logits))

    def test_real_data_batch_forward_smoke(self) -> None:
        import torch

        from torch.utils.data import DataLoader

        from Source.Model import (
            BrushWrightStrokeDataset,
            BrushWrightStrokePredictor,
            DraftImageEncoderConfig,
            DraftImageEncoder,
            StrokeEncoderConfig,
            collate_stroke_chunks,
        )
        from Source.Model.stroke_decoder import StrokeChunkDecoderConfig

        if not Path("Data/Train/dataset_manifest.json").exists():
            self.skipTest("generated Data/Train dataset is not present")
        dataset = BrushWrightStrokeDataset("Data/Train", chunk_size=64, max_base_strokes=192, require_v1_contract=False)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_stroke_chunks)
        batch = next(iter(loader))
        model = BrushWrightStrokePredictor(
            encoder_config=StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=192),
            decoder_config=StrokeChunkDecoderConfig(
                model_dim=32,
                num_layers=1,
                num_heads=4,
                ff_dim=64,
                dropout=0.0,
                chunk_size=64,
                max_chunks=16,
                query_mode="spatial",
                spatial_grid_size=2,
            ),
            image_encoder_config=DraftImageEncoderConfig(model_dim=32, hidden_dim=16, grid_size=2, dropout=0.0, input_channels=3),
        )
        output = model(batch)

        self.assertEqual(output.pred_numeric.shape, (2, 64, 9))
        self.assertEqual(output.pred_brush_logits.shape, (2, 64, dataset.tokenizer.vocab_size))

        encoder = DraftImageEncoder(DraftImageEncoderConfig(model_dim=32, hidden_dim=16, grid_size=2, dropout=0.0, input_channels=3))
        image_output = encoder(batch.draft_images)
        self.assertEqual(image_output.features.shape, (2, 4, 32))


if __name__ == "__main__":
    unittest.main()
