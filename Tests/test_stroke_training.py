from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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


def _program(strokes: list[dict]) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {},
        "strokes": strokes,
    }


def _write_sample(split_root: Path, sample_id: str, base_count: int, finishing_count: int) -> None:
    sample_dir = split_root / sample_id
    sample_dir.mkdir(parents=True)
    _write_json(
        sample_dir / "sample.json",
        {
            "version": 1,
            "sample_id": sample_id,
            "base_count": base_count,
            "finishing_count": finishing_count,
            "stroke_count_adjusted": False,
            "base_strokes": "base_strokes.json",
            "finishing_strokes": "finishing_strokes.json",
            "draft_image": "draft.png",
            "finished_image": "finished.png",
        },
    )
    _write_json(sample_dir / "base_strokes.json", _program([_stroke(index) for index in range(base_count)]))
    _write_json(
        sample_dir / "finishing_strokes.json",
        _program([_stroke(index) for index in range(finishing_count)]),
    )
    _write_image(sample_dir / "draft.png", (64, 96, 128))
    _write_image(sample_dir / "finished.png", (96, 128, 160))


def _write_manifest(split_root: Path, sample_ids: list[str]) -> None:
    _write_json(
        split_root / "dataset_manifest.json",
        {
            "version": 1,
            "split": split_root.name,
            "sample_count": len(sample_ids),
            "samples": [{"sample_id": sample_id, "path": sample_id} for sample_id in sample_ids],
        },
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (16, 16), color).save(path)


def _write_tiny_data_root(root: Path) -> None:
    for split in ("Train", "Val"):
        split_root = root / split
        split_root.mkdir(parents=True)
        _write_sample(split_root, "sample_a", base_count=4, finishing_count=12)
        _write_sample(split_root, "sample_b", base_count=3, finishing_count=8)
        _write_manifest(split_root, ["sample_a", "sample_b"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for stroke training tests")
class StrokeTrainingTest(unittest.TestCase):
    def test_loss_ignores_padded_target_slots(self) -> None:
        import torch

        from Source.Model import StrokeBatch, StrokePredictionOutput, StrokeTokenBatch, compute_stroke_loss

        prediction = StrokePredictionOutput(
            pred_numeric=torch.zeros(1, 4, 9),
            pred_brush_logits=torch.tensor(
                [
                    [
                        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
        )
        batch = _fake_batch(target_numeric=torch.zeros(1, 4, 9), target_brush_ids=torch.tensor([[0, 1, 2, 3]]))
        changed_batch = _fake_batch(
            target_numeric=torch.tensor([[[0.0] * 9, [0.0] * 9, [99.0] * 9, [99.0] * 9]]),
            target_brush_ids=torch.tensor([[0, 1, 6, 7]]),
        )

        first = compute_stroke_loss(prediction, batch)
        second = compute_stroke_loss(prediction, changed_batch)

        self.assertEqual(first.valid_target_count, 2)
        self.assertTrue(torch.equal(first.total, second.total))

    def test_set_matched_loss_is_order_invariant(self) -> None:
        import torch

        from Source.Model import StrokePredictionOutput, compute_stroke_loss

        stroke_a = torch.tensor([0.10, 0.20, 0.10, 0.12, 0.04, 1.0, 0.2, 0.3, 0.4])
        stroke_b = torch.tensor([0.80, 0.70, 0.90, 0.20, 0.08, 1.0, 0.7, 0.6, 0.5])
        pred_numeric = torch.stack([stroke_a, stroke_b, torch.zeros(9), torch.zeros(9)]).unsqueeze(0)
        pred_brush_logits = torch.zeros(1, 4, 8)
        pred_brush_logits[0, 0, 2] = 5.0
        pred_brush_logits[0, 1, 3] = 5.0
        prediction = StrokePredictionOutput(pred_numeric=pred_numeric, pred_brush_logits=pred_brush_logits)

        ordered_batch = _fake_batch(
            target_numeric=pred_numeric.clone(),
            target_brush_ids=torch.tensor([[2, 3, 0, 0]]),
            target_padding_mask=torch.tensor([[False, False, True, True]]),
        )
        shuffled_batch = _fake_batch(
            target_numeric=torch.stack([stroke_b, stroke_a, torch.zeros(9), torch.zeros(9)]).unsqueeze(0),
            target_brush_ids=torch.tensor([[3, 2, 0, 0]]),
            target_padding_mask=torch.tensor([[False, False, True, True]]),
        )

        ordered = compute_stroke_loss(
            prediction,
            ordered_batch,
            distribution_weight=0.0,
            visual_weight=0.0,
            set_matching=True,
        )
        shuffled = compute_stroke_loss(
            prediction,
            shuffled_batch,
            distribution_weight=0.0,
            visual_weight=0.0,
            set_matching=True,
        )

        self.assertAlmostEqual(float(ordered.numeric), float(shuffled.numeric), places=6)
        self.assertAlmostEqual(float(ordered.brush), float(shuffled.brush), places=6)
        self.assertAlmostEqual(float(ordered.total), float(shuffled.total), places=6)

    def test_ordered_loss_penalizes_shuffled_targets(self) -> None:
        import torch

        from Source.Model import StrokePredictionOutput, compute_stroke_loss

        stroke_a = torch.tensor([0.10, 0.20, 0.10, 0.12, 0.04, 1.0, 0.2, 0.3, 0.4])
        stroke_b = torch.tensor([0.80, 0.70, 0.90, 0.20, 0.08, 1.0, 0.7, 0.6, 0.5])
        pred_numeric = torch.stack([stroke_a, stroke_b, torch.zeros(9), torch.zeros(9)]).unsqueeze(0)
        prediction = StrokePredictionOutput(pred_numeric=pred_numeric, pred_brush_logits=torch.zeros(1, 4, 8))
        ordered_batch = _fake_batch(
            target_numeric=pred_numeric.clone(),
            target_brush_ids=torch.tensor([[0, 0, 0, 0]]),
            target_padding_mask=torch.tensor([[False, False, True, True]]),
        )
        shuffled_batch = _fake_batch(
            target_numeric=torch.stack([stroke_b, stroke_a, torch.zeros(9), torch.zeros(9)]).unsqueeze(0),
            target_brush_ids=torch.tensor([[0, 0, 0, 0]]),
            target_padding_mask=torch.tensor([[False, False, True, True]]),
        )

        ordered = compute_stroke_loss(
            prediction,
            ordered_batch,
            brush_weight=0.0,
            distribution_weight=0.0,
            visual_weight=0.0,
            set_matching=False,
        )
        shuffled = compute_stroke_loss(
            prediction,
            shuffled_batch,
            brush_weight=0.0,
            distribution_weight=0.0,
            visual_weight=0.0,
            set_matching=False,
        )

        self.assertLess(float(ordered.total), 1e-6)
        self.assertGreater(float(shuffled.total), float(ordered.total))

    def test_set_matching_ignores_padded_targets(self) -> None:
        import torch

        from Source.Model import match_stroke_targets

        stroke_a = torch.tensor([0.10, 0.20, 0.10, 0.12, 0.04, 1.0, 0.2, 0.3, 0.4])
        stroke_b = torch.tensor([0.80, 0.70, 0.90, 0.20, 0.08, 1.0, 0.7, 0.6, 0.5])
        pred_numeric = torch.stack([stroke_b, stroke_a, torch.zeros(9), torch.zeros(9)]).unsqueeze(0)
        target_numeric = torch.stack([stroke_a, torch.full((9,), 99.0), stroke_b, torch.zeros(9)]).unsqueeze(0)
        matched_numeric, matched_brush_ids, matched_padding_mask = match_stroke_targets(
            pred_numeric,
            target_numeric,
            torch.tensor([[2, 7, 3, 0]]),
            torch.tensor([[False, True, False, True]]),
        )

        self.assertEqual(int((~matched_padding_mask).sum()), 2)
        self.assertTrue(torch.allclose(matched_numeric[0, 0], stroke_b))
        self.assertTrue(torch.allclose(matched_numeric[0, 1], stroke_a))
        self.assertEqual(int(matched_brush_ids[0, 0]), 3)
        self.assertEqual(int(matched_brush_ids[0, 1]), 2)

    def test_loss_returns_finite_scalar_components(self) -> None:
        import torch

        from Source.Model import StrokePredictionOutput, compute_stroke_loss

        prediction = StrokePredictionOutput(
            pred_numeric=torch.full((1, 4, 9), 0.5),
            pred_brush_logits=torch.zeros(1, 4, 8),
        )
        loss = compute_stroke_loss(
            prediction,
            _fake_batch(target_numeric=torch.zeros(1, 4, 9), target_brush_ids=torch.zeros(1, 4, dtype=torch.long)),
        )

        self.assertEqual(loss.total.ndim, 0)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(torch.isfinite(loss.numeric))
        self.assertTrue(torch.isfinite(loss.brush))
        self.assertTrue(torch.isfinite(loss.distribution))
        self.assertTrue(torch.isfinite(loss.visual))

    def test_render_loss_is_lower_for_matching_render(self) -> None:
        import torch

        from Source.Model import StrokeBatch, StrokeTokenBatch, compute_render_loss, render_paint_transformer_soft_strokes

        target_numeric = torch.tensor(
            [
                [
                    [0.50, 0.50, 0.25, 0.20, 0.08, 1.0, 0.9, 0.1, 0.1],
                    [0.25, 0.25, 0.75, 0.12, 0.05, 1.0, 0.1, 0.8, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        draft = torch.zeros(1, 3, 64, 64)
        present_logits = torch.full((1, 2), 20.0)
        goal = render_paint_transformer_soft_strokes(draft, target_numeric, present_logits).detach()
        batch = StrokeBatch(
            base_tokens=StrokeTokenBatch(
                numeric=torch.zeros(1, 1, 9),
                brush_ids=torch.tensor([[2]]),
                padding_mask=torch.tensor([[False]]),
                lengths=torch.tensor([1]),
            ),
            target_numeric=target_numeric,
            target_brush_ids=torch.tensor([[2, 2]]),
            target_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            sample_ids=("sample_a",),
            chunk_starts=torch.tensor([0]),
            chunk_ends=torch.tensor([2]),
            stroke_count_adjusted=torch.tensor([False]),
            draft_images=draft,
            goal_images=goal,
            error_maps=None,
        )

        matching = compute_render_loss(target_numeric, batch)
        shifted_numeric = target_numeric.clone()
        shifted_numeric[..., 0] = (shifted_numeric[..., 0] + 0.25).clamp(max=1.0)
        shifted = compute_render_loss(shifted_numeric, batch)

        self.assertLess(matching.item(), 1e-5)
        self.assertGreater(shifted.item(), matching.item())

    def test_changed_region_render_loss_is_lower_for_matching_delta(self) -> None:
        import torch

        from Source.Model import StrokeBatch, StrokeTokenBatch, compute_render_loss, render_paint_transformer_soft_strokes

        target_numeric = torch.tensor(
            [
                [
                    [0.50, 0.50, 0.25, 0.20, 0.08, 1.0, 0.9, 0.1, 0.1],
                    [0.25, 0.25, 0.75, 0.12, 0.05, 1.0, 0.1, 0.8, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        draft = torch.zeros(1, 3, 64, 64)
        present_logits = torch.full((1, 2), 20.0)
        goal = render_paint_transformer_soft_strokes(draft, target_numeric, present_logits).detach()
        batch = StrokeBatch(
            base_tokens=StrokeTokenBatch(
                numeric=torch.zeros(1, 1, 9),
                brush_ids=torch.tensor([[2]]),
                padding_mask=torch.tensor([[False]]),
                lengths=torch.tensor([1]),
            ),
            target_numeric=target_numeric,
            target_brush_ids=torch.tensor([[2, 2]]),
            target_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            sample_ids=("sample_a",),
            chunk_starts=torch.tensor([0]),
            chunk_ends=torch.tensor([2]),
            stroke_count_adjusted=torch.tensor([False]),
            draft_images=draft,
            goal_images=goal,
            error_maps=torch.abs(goal - draft),
        )

        matching = compute_render_loss(target_numeric, batch)
        shifted_numeric = target_numeric.clone()
        shifted_numeric[..., 0] = (shifted_numeric[..., 0] + 0.25).clamp(max=1.0)
        shifted = compute_render_loss(shifted_numeric, batch)

        self.assertTrue(torch.isfinite(matching))
        self.assertTrue(torch.isfinite(shifted))
        self.assertLess(matching.item(), 1e-5)
        self.assertGreater(shifted.item(), matching.item())

    def test_distribution_loss_penalizes_collapsed_prediction(self) -> None:
        import torch

        from Source.Model import compute_distribution_loss

        target = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.5, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.5, 0.10, 0.10, 1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5, 0.05, 0.05, 1.0, 0.5, 0.5, 0.5],
                    [0.2, 0.8, 0.5, 0.08, 0.02, 1.0, 0.2, 0.8, 0.4],
                ]
            ],
            dtype=torch.float32,
        )
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)
        matching = compute_distribution_loss(target, target, padding_mask)
        collapsed = compute_distribution_loss(torch.full_like(target, 0.5), target, padding_mask)

        self.assertLess(matching.item(), 1e-6)
        self.assertGreater(collapsed.item(), matching.item())

    def test_one_optimizer_step_changes_model_parameter(self) -> None:
        import torch

        from Source.Model import (
            BrushWrightStrokePredictor,
            DraftImageEncoderConfig,
            StrokeEncoderConfig,
            StrokePredictionOutput,
            compute_stroke_loss,
        )
        from Source.Model.stroke_decoder import StrokeChunkDecoderConfig

        torch.manual_seed(123)
        batch = _fake_batch(
            target_numeric=torch.full((1, 4, 9), 0.25),
            target_brush_ids=torch.full((1, 4), 2, dtype=torch.long),
            target_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        )
        model = BrushWrightStrokePredictor(
            encoder_config=StrokeEncoderConfig(model_dim=16, num_layers=1, num_heads=4, ff_dim=32, dropout=0.0, max_strokes=2),
            decoder_config=StrokeChunkDecoderConfig(model_dim=16, num_layers=1, num_heads=4, ff_dim=32, dropout=0.0, chunk_size=4, max_chunks=3),
            image_encoder_config=DraftImageEncoderConfig(model_dim=16, hidden_dim=16, grid_size=2, dropout=0.0),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        prediction = model(batch)
        self.assertIsInstance(prediction, StrokePredictionOutput)
        loss = compute_stroke_loss(prediction, batch)
        loss.total.backward()
        optimizer.step()

        changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters()))
        self.assertTrue(changed)

    def test_overfit_dataset_limiting_uses_requested_number_of_chunks(self) -> None:
        from Source.Model.train_strokes import StrokeTrainingConfig, train_strokes

        with tempfile.TemporaryDirectory() as root_name:
            data_root = Path(root_name) / "Data"
            output_dir = Path(root_name) / "Checkpoints"
            _write_tiny_data_root(data_root)

            result = train_strokes(
                StrokeTrainingConfig(
                    data_root=data_root,
                    output_dir=output_dir,
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    model_dim=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    num_workers=0,
                    dropout=0.0,
                    chunk_size=4,
                    max_base_strokes=4,
                    max_chunks=3,
                    image_grid_size=2,
                    overfit_samples=3,
                    train_repeat_factor=2,
                    visual_validation_samples=0,
                    require_v1_contract=False,
                    visual_weight=0.0,
                )
            )

            self.assertEqual(result["train_chunk_count"], 6)
            self.assertEqual(result["val_chunk_count"], 3)

    def test_checkpoint_save_writes_expected_keys(self) -> None:
        import torch

        from Source.Model.train_strokes import StrokeTrainingConfig, train_strokes

        with tempfile.TemporaryDirectory() as root_name:
            data_root = Path(root_name) / "Data"
            output_dir = Path(root_name) / "Checkpoints"
            _write_tiny_data_root(data_root)

            train_strokes(
                StrokeTrainingConfig(
                    data_root=data_root,
                    output_dir=output_dir,
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    model_dim=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    num_workers=0,
                    dropout=0.0,
                    chunk_size=4,
                    max_base_strokes=4,
                    max_chunks=3,
                    image_grid_size=2,
                    overfit_samples=2,
                    visual_validation_samples=0,
                    require_v1_contract=False,
                    visual_weight=0.0,
                )
            )
            checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")

            self.assertEqual(
                set(checkpoint),
                {
                    "model_state_dict",
                    "optimizer_state_dict",
                    "epoch",
                    "metrics",
                    "encoder_config",
                    "decoder_config",
                    "image_encoder_config",
                    "checkpoint_type",
                    "global_step",
                    "metrics_log",
                    "tokenizer",
                },
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["checkpoint_type"], "epoch")
            self.assertGreaterEqual(checkpoint["global_step"], 1)
            self.assertIn("train", checkpoint["metrics"])
            self.assertIn("val", checkpoint["metrics"])
            self.assertIn("brush_to_id", checkpoint["tokenizer"])

    def test_resume_checkpoint_continues_training(self) -> None:
        import torch

        from Source.Model.train_strokes import StrokeTrainingConfig, train_strokes

        with tempfile.TemporaryDirectory() as root_name:
            data_root = Path(root_name) / "Data"
            output_dir = Path(root_name) / "Checkpoints"
            _write_tiny_data_root(data_root)
            base_config = StrokeTrainingConfig(
                data_root=data_root,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                device="cpu",
                model_dim=16,
                encoder_layers=1,
                decoder_layers=1,
                num_heads=4,
                ff_dim=32,
                num_workers=0,
                dropout=0.0,
                chunk_size=4,
                max_base_strokes=4,
                max_chunks=3,
                image_grid_size=2,
                overfit_samples=2,
                checkpoint_every_steps=1,
                visual_validation_samples=0,
                require_v1_contract=False,
                visual_weight=0.0,
            )
            first = train_strokes(base_config)
            self.assertEqual(first["epochs"][-1]["epoch"], 1)
            self.assertTrue((output_dir / "step_latest.pt").exists())

            resumed = train_strokes(
                StrokeTrainingConfig(
                    data_root=data_root,
                    output_dir=output_dir,
                    epochs=2,
                    batch_size=2,
                    device="cpu",
                    model_dim=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    num_workers=0,
                    dropout=0.0,
                    chunk_size=4,
                    max_base_strokes=4,
                    max_chunks=3,
                    image_grid_size=2,
                    overfit_samples=2,
                    resume_checkpoint=output_dir / "latest.pt",
                    visual_validation_samples=0,
                    require_v1_contract=False,
                    visual_weight=0.0,
                )
            )
            checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")

            self.assertEqual(resumed["epochs"][-1]["epoch"], 2)
            self.assertEqual(checkpoint["epoch"], 2)


def _fake_batch(
    target_numeric,
    target_brush_ids,
    target_padding_mask=None,
    with_goal: bool = True,
):
    import torch

    from Source.Model import StrokeBatch, StrokeTokenBatch

    if target_padding_mask is None:
        target_padding_mask = torch.tensor([[False, False, True, True]])
    draft_images = torch.zeros(1, 3, 64, 64)
    goal_images = torch.ones(1, 3, 64, 64) if with_goal else None
    error_maps = torch.abs(goal_images - draft_images) if goal_images is not None else None
    return StrokeBatch(
        base_tokens=StrokeTokenBatch(
            numeric=torch.zeros(1, 2, 9),
            brush_ids=torch.tensor([[2, 0]]),
            padding_mask=torch.tensor([[False, True]]),
            lengths=torch.tensor([1]),
        ),
        target_numeric=target_numeric.float(),
        target_brush_ids=target_brush_ids.long(),
        target_padding_mask=target_padding_mask,
        sample_ids=("sample_a",),
        chunk_starts=torch.tensor([0]),
        chunk_ends=torch.tensor([int((~target_padding_mask[0]).sum().item())]),
        stroke_count_adjusted=torch.tensor([False]),
        draft_images=draft_images,
        goal_images=goal_images,
        error_maps=error_maps,
    )


if __name__ == "__main__":
    unittest.main()
