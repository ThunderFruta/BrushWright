from __future__ import annotations

import importlib.util
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
    value = index / 100.0
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


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model tests")
class StrokeEncoderTest(unittest.TestCase):
    def test_tokenizer_emits_expected_shape_and_brush_ids(self) -> None:
        import torch

        from Source.Model import PAD_BRUSH_ID, StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=4)
        batch = tokenizer.encode_program(_program([_stroke(1), _stroke(2, brush="flat_oil")]))

        self.assertEqual(batch.numeric.shape, (1, 4, 9))
        self.assertEqual(batch.brush_ids.shape, (1, 4))
        self.assertEqual(batch.padding_mask.shape, (1, 4))
        self.assertEqual(batch.lengths.tolist(), [2])
        self.assertEqual(batch.brush_ids[0, 0].item(), tokenizer.brush_to_id["paint_transformer_rect"])
        self.assertEqual(batch.brush_ids[0, 1].item(), tokenizer.brush_to_id["flat_oil"])
        self.assertEqual(batch.brush_ids[0, 2].item(), PAD_BRUSH_ID)
        self.assertTrue(torch.allclose(batch.numeric[0, 0], torch.tensor([0.01, 0.25, 0.5, 0.1, 0.02, 1.0, 0.2, 0.3, 0.4])))

    def test_tokenizer_padding_mask_for_mixed_length_batch(self) -> None:
        from Source.Model import StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=4)
        batch = tokenizer.encode_programs(
            [
                _program([_stroke(1)]),
                _program([_stroke(2), _stroke(3), _stroke(4)]),
            ]
        )

        self.assertEqual(batch.lengths.tolist(), [1, 3])
        self.assertEqual(batch.padding_mask.tolist(), [[False, True, True, True], [False, False, False, True]])

    def test_tokenizer_maps_unknown_brush_to_unk(self) -> None:
        from Source.Model import UNK_BRUSH_ID, StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=2)
        batch = tokenizer.encode_program(_program([_stroke(1, brush="custom_brush")]))

        self.assertEqual(batch.brush_ids[0, 0].item(), UNK_BRUSH_ID)

    def test_tokenizer_rejects_overlong_program_without_truncation(self) -> None:
        from Source.Model import StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=2)
        with self.assertRaisesRegex(ValueError, "max_strokes"):
            tokenizer.encode_program(_program([_stroke(1), _stroke(2), _stroke(3)]))

    def test_encoder_forward_shapes_for_single_and_batch(self) -> None:
        from Source.Model import StrokeEncoder, StrokeEncoderConfig, StrokeTokenizer

        tokenizer = StrokeTokenizer(max_strokes=4)
        encoder = StrokeEncoder(
            StrokeEncoderConfig(model_dim=32, num_layers=2, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4)
        )
        single = tokenizer.encode_program(_program([_stroke(1), _stroke(2)]))
        single_output = encoder(single.numeric, single.brush_ids, single.padding_mask)

        self.assertEqual(single_output.features.shape, (1, 4, 32))
        self.assertEqual(single_output.pooled.shape, (1, 32))
        self.assertEqual(single_output.padding_mask.shape, (1, 4))

        mixed = tokenizer.encode_programs(
            [
                _program([_stroke(1)]),
                _program([_stroke(2), _stroke(3)]),
            ]
        )
        mixed_output = encoder(mixed.numeric, mixed.brush_ids, mixed.padding_mask)

        self.assertEqual(mixed_output.features.shape, (2, 4, 32))
        self.assertEqual(mixed_output.pooled.shape, (2, 32))

    def test_encoder_is_deterministic_in_eval_mode(self) -> None:
        import torch

        from Source.Model import StrokeEncoder, StrokeEncoderConfig, StrokeTokenizer

        torch.manual_seed(123)
        tokenizer = StrokeTokenizer(max_strokes=4)
        encoder = StrokeEncoder(
            StrokeEncoderConfig(model_dim=32, num_layers=2, num_heads=4, ff_dim=64, dropout=0.1, max_strokes=4)
        )
        encoder.eval()
        batch = tokenizer.encode_program(_program([_stroke(1), _stroke(2), _stroke(3)]))

        with torch.no_grad():
            first = encoder(batch.numeric, batch.brush_ids, batch.padding_mask)
            second = encoder(batch.numeric, batch.brush_ids, batch.padding_mask)

        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.pooled, second.pooled))


if __name__ == "__main__":
    unittest.main()
