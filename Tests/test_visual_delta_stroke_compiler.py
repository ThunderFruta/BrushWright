from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _stroke(x: float, y: float, brush: str = "paint_transformer_rect") -> dict:
    return {
        "x": x,
        "y": y,
        "angle": 0.5,
        "length": 0.02,
        "width": 0.01,
        "color": [0.8, 0.7, 0.6],
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


def _write_sample(
    split_root: Path,
    sample_id: str = "sample_a",
    finishing_strokes: list[dict] | None = None,
    structure_target: bool = False,
    target_contract: str | None = None,
) -> None:
    sample_dir = split_root / sample_id
    sample_dir.mkdir(parents=True)
    sample = {
        "version": 1,
        "sample_id": sample_id,
        "base_count": 1,
        "finishing_count": 2,
        "stroke_count_adjusted": False,
        "base_strokes": "base_strokes.json",
        "finishing_strokes": "finishing_strokes.json",
        "draft_image": "draft.png",
        "finished_image": "finished.png",
        "split_manifest": "split_manifest.json",
    }
    split_manifest = {"version": 1}
    if target_contract is not None:
        sample["target_contract"] = target_contract
        split_manifest["target_contract"] = target_contract
    if structure_target:
        sample["target_selection_mode"] = "structure_first_v1"
        sample["target_selection_manifest"] = "target_selection_manifest.json"
        _write_json(
            sample_dir / "target_selection_manifest.json",
            {
                "version": 1,
                "target_selection_mode": "structure_first_v1",
                "selected_source_indices": [1, 2],
            },
        )
    _write_json(sample_dir / "sample.json", sample)
    _write_json(sample_dir / "split_manifest.json", split_manifest)
    _write_json(sample_dir / "base_strokes.json", _program([_stroke(0.1, 0.1)]))
    _write_json(
        sample_dir / "finishing_strokes.json",
        _program(finishing_strokes if finishing_strokes is not None else [_stroke(0.2, 0.2), _stroke(0.8, 0.8)]),
    )
    _write_images(sample_dir / "draft.png", sample_dir / "finished.png")


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


def _write_images(draft_path: Path, finished_path: Path) -> None:
    from PIL import Image, ImageDraw

    draft = Image.new("RGB", (512, 512), (32, 32, 32))
    finished = draft.copy()
    draw = ImageDraw.Draw(finished)
    draw.rectangle((64, 64, 160, 160), fill=(220, 180, 120))
    draw.rectangle((384, 384, 480, 480), fill=(220, 180, 120))
    draft.save(draft_path)
    finished.save(finished_path)


def _write_data_root(root: Path) -> None:
    for split in ("Train", "Val", "Test"):
        split_root = root / split
        split_root.mkdir(parents=True)
        _write_sample(split_root, "sample_a")
        _write_manifest(split_root, ["sample_a"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for V3 tests")
class VisualDeltaStrokeCompilerTest(unittest.TestCase):
    def test_dataset_patch_tensor_and_targets(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset
        from Source.Model.visual_delta_dataset import DEFAULT_PATCH_SIZE, DEFAULT_PATCH_STRIDE

        self.assertEqual(DEFAULT_PATCH_SIZE, 64)
        self.assertEqual(DEFAULT_PATCH_STRIDE, 64)

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root)
            _write_manifest(split_root, ["sample_a"])

            dataset = VisualDeltaStrokeDataset(split_root, patch_size=128, patch_stride=128, negative_patch_ratio=0.0)
            item = dataset[0]

            self.assertEqual(item.patch_tensor.shape, (10, 128, 128))
            self.assertGreater(float(item.patch_tensor[9].sum()), 0.0)
            self.assertEqual(item.target_numeric.shape, (256, 9))
            self.assertEqual(item.target_present.shape, (256,))
            self.assertGreaterEqual(int(item.target_present.sum().item()), 1)
            self.assertFalse(bool(item.target_padding_mask[0]))

    def test_default_patch_size_keeps_targets_under_one_shot_limit(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root)
            _write_manifest(split_root, ["sample_a"])

            dataset = VisualDeltaStrokeDataset(split_root, negative_patch_ratio=0.0)
            item = dataset[0]

            self.assertEqual(item.patch_tensor.shape, (10, 64, 64))
            self.assertLessEqual(int(item.target_present.sum().item()), 256)

    def test_patch_numeric_round_trips_to_global_stroke(self) -> None:
        from Source.Model import patch_numeric_to_global_stroke

        stroke = patch_numeric_to_global_stroke(
            [0.5, 0.5, 0.25, 0.08, 0.04, 1.0, 0.2, 0.3, 0.4],
            "paint_transformer_rect",
            [0.25, 0.25, 0.5, 0.5],
        )

        self.assertAlmostEqual(stroke["x"], 0.375)
        self.assertAlmostEqual(stroke["y"], 0.375)
        self.assertAlmostEqual(stroke["length"], 0.02)
        self.assertEqual(stroke["brush"], "paint_transformer_rect")

    def test_model_forward_shapes_and_ranges(self) -> None:
        import torch

        from Source.Model import (
            VisualDeltaStrokeCompiler,
            VisualDeltaStrokeCompilerConfig,
            VisualDeltaStrokeDataset,
            collate_visual_delta_patches,
        )

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root)
            _write_manifest(split_root, ["sample_a"])
            dataset = VisualDeltaStrokeDataset(split_root, patch_size=128, patch_stride=128, negative_patch_ratio=0.0)
            batch = collate_visual_delta_patches([dataset[0]])
            model = VisualDeltaStrokeCompiler(
                VisualDeltaStrokeCompilerConfig(
                    model_dim=16,
                    hidden_dim=8,
                    num_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    dropout=0.0,
                    grid_size=2,
                    max_strokes=4,
                    coarse_grid_size=1,
                    detail_grid_rows=1,
                    detail_grid_cols=3,
                )
            )
            batch = type(batch)(
                patch_tensor=batch.patch_tensor,
                target_numeric=batch.target_numeric[:, :4],
                target_brush_ids=batch.target_brush_ids[:, :4],
                target_present=batch.target_present[:, :4],
                target_padding_mask=batch.target_padding_mask[:, :4],
                sample_ids=batch.sample_ids,
                patch_bounds=batch.patch_bounds,
                changed=batch.changed,
            )
            output = model(batch)

            self.assertEqual(output.pred_numeric.shape, (1, 4, 9))
            self.assertEqual(output.pred_brush_logits.shape, (1, 4, 8))
            self.assertEqual(output.pred_present_logits.shape, (1, 4))
            self.assertTrue(torch.all(output.pred_numeric[..., 0:2] >= 0.0))
            self.assertTrue(torch.all(output.pred_numeric[..., 0:2] <= 1.0))
            render_area = output.pred_numeric[..., 3] * output.pred_numeric[..., 4] * 128 * 128
            self.assertTrue(torch.all(render_area > 8.0))

    def test_loss_and_optimizer_step(self) -> None:
        import torch

        from Source.Model import (
            VisualDeltaStrokeCompiler,
            VisualDeltaStrokeCompilerConfig,
            VisualDeltaStrokeDataset,
            collate_visual_delta_patches,
            compute_visual_delta_loss,
        )

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root)
            _write_manifest(split_root, ["sample_a"])
            dataset = VisualDeltaStrokeDataset(split_root, patch_size=128, patch_stride=128, max_strokes_per_patch=4, negative_patch_ratio=0.0)
            batch = collate_visual_delta_patches([dataset[0]])
            model = VisualDeltaStrokeCompiler(
                VisualDeltaStrokeCompilerConfig(
                    model_dim=16,
                    hidden_dim=8,
                    num_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    dropout=0.0,
                    grid_size=2,
                    max_strokes=4,
                    coarse_grid_size=1,
                    detail_grid_rows=1,
                    detail_grid_cols=3,
                )
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            before = [parameter.detach().clone() for parameter in model.parameters()]

            loss = compute_visual_delta_loss(model(batch), batch)
            loss.total.backward()
            optimizer.step()

            self.assertTrue(torch.isfinite(loss.total))
            self.assertGreaterEqual(loss.valid_target_count, 1)
            self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())))

    def test_set_matched_loss_is_target_order_invariant(self) -> None:
        import torch

        from Source.Model import VisualDeltaBatch, VisualDeltaPredictionOutput, compute_visual_delta_loss

        patch_tensor = torch.zeros(1, 10, 64, 64)
        pred_numeric = torch.tensor(
            [
                [
                    [0.20, 0.20, 0.5, 0.05, 0.04, 1.0, 0.8, 0.7, 0.6],
                    [0.80, 0.80, 0.5, 0.05, 0.04, 1.0, 0.2, 0.3, 0.4],
                ]
            ],
            dtype=torch.float32,
        )
        prediction = VisualDeltaPredictionOutput(
            pred_numeric=pred_numeric,
            pred_brush_logits=torch.zeros(1, 2, 8),
            pred_present_logits=torch.full((1, 2), 4.0),
        )
        first_batch = VisualDeltaBatch(
            patch_tensor=patch_tensor,
            target_numeric=pred_numeric.clone(),
            target_brush_ids=torch.zeros(1, 2, dtype=torch.long),
            target_present=torch.ones(1, 2),
            target_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            sample_ids=("sample_a",),
            patch_bounds=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            changed=torch.tensor([True]),
        )
        second_batch = type(first_batch)(
            patch_tensor=first_batch.patch_tensor,
            target_numeric=pred_numeric.flip(1).clone(),
            target_brush_ids=first_batch.target_brush_ids,
            target_present=first_batch.target_present,
            target_padding_mask=first_batch.target_padding_mask,
            sample_ids=first_batch.sample_ids,
            patch_bounds=first_batch.patch_bounds,
            changed=first_batch.changed,
        )

        first = compute_visual_delta_loss(prediction, first_batch, slot_aware_targets=False)
        second = compute_visual_delta_loss(prediction, second_batch, slot_aware_targets=False)

        self.assertAlmostEqual(float(first.numeric), float(second.numeric), places=6)
        self.assertAlmostEqual(float(first.total), float(second.total), places=6)

    def test_slot_aware_assignment_maps_targets_to_nearest_compatible_slots(self) -> None:
        import torch

        from Source.Model import match_visual_delta_strokes_slot_aware

        target_numeric = torch.zeros(1, 4, 9)
        target_numeric[0, 0] = torch.tensor([0.82, 0.50, 0.3, 0.05, 0.04, 1.0, 0.2, 0.3, 0.4])
        target_numeric[0, 1] = torch.tensor([0.52, 0.52, 0.3, 0.30, 0.15, 1.0, 0.5, 0.5, 0.5])
        target_brush_ids = torch.tensor([[2, 3, 0, 0]])
        target_padding_mask = torch.tensor([[False, False, True, True]])

        matched_numeric, matched_brush_ids, matched_mask = match_visual_delta_strokes_slot_aware(
            target_numeric,
            target_brush_ids,
            target_padding_mask,
        )

        self.assertTrue(bool(matched_mask[0, 0]))
        self.assertTrue(bool(matched_mask[0, 3]))
        self.assertEqual(int(matched_brush_ids[0, 0]), 3)
        self.assertEqual(int(matched_brush_ids[0, 3]), 2)
        self.assertAlmostEqual(float(matched_numeric[0, 3, 0]), 0.82)

    def test_unmatched_predictions_are_penalized_by_present_loss(self) -> None:
        import torch

        from Source.Model import VisualDeltaBatch, VisualDeltaPredictionOutput, compute_visual_delta_loss

        batch = VisualDeltaBatch(
            patch_tensor=torch.zeros(1, 10, 64, 64),
            target_numeric=torch.zeros(1, 2, 9),
            target_brush_ids=torch.zeros(1, 2, dtype=torch.long),
            target_present=torch.zeros(1, 2),
            target_padding_mask=torch.ones(1, 2, dtype=torch.bool),
            sample_ids=("sample_a",),
            patch_bounds=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            changed=torch.tensor([False]),
        )
        numeric = torch.zeros(1, 2, 9)
        high_present = VisualDeltaPredictionOutput(
            pred_numeric=numeric,
            pred_brush_logits=torch.zeros(1, 2, 8),
            pred_present_logits=torch.full((1, 2), 4.0),
        )
        low_present = VisualDeltaPredictionOutput(
            pred_numeric=numeric,
            pred_brush_logits=torch.zeros(1, 2, 8),
            pred_present_logits=torch.full((1, 2), -4.0),
        )

        self.assertGreater(
            float(compute_visual_delta_loss(high_present, batch).present),
            float(compute_visual_delta_loss(low_present, batch).present),
        )

    def test_present_positive_weight_penalizes_missing_target_strokes(self) -> None:
        import torch

        from Source.Model import VisualDeltaBatch, VisualDeltaPredictionOutput, compute_visual_delta_loss

        batch = VisualDeltaBatch(
            patch_tensor=torch.zeros(1, 10, 64, 64),
            target_numeric=torch.tensor(
                [[[0.5, 0.5, 0.2, 0.02, 0.01, 1.0, 0.7, 0.6, 0.5], [0.0] * 9]],
                dtype=torch.float32,
            ),
            target_brush_ids=torch.zeros(1, 2, dtype=torch.long),
            target_present=torch.tensor([[1.0, 0.0]]),
            target_padding_mask=torch.tensor([[False, True]]),
            sample_ids=("sample_a",),
            patch_bounds=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            changed=torch.tensor([True]),
        )
        missing_target = VisualDeltaPredictionOutput(
            pred_numeric=torch.zeros(1, 2, 9),
            pred_brush_logits=torch.zeros(1, 2, 8),
            pred_present_logits=torch.full((1, 2), -4.0),
        )

        unweighted = compute_visual_delta_loss(
            missing_target,
            batch,
            present_positive_weight=1.0,
        )
        weighted = compute_visual_delta_loss(
            missing_target,
            batch,
            present_positive_weight=8.0,
        )

        self.assertGreater(float(weighted.present), float(unweighted.present))

    def test_empty_patch_slot_aware_targets_train_present_off(self) -> None:
        import torch

        from Source.Model import VisualDeltaBatch, VisualDeltaPredictionOutput, compute_visual_delta_loss

        batch = VisualDeltaBatch(
            patch_tensor=torch.zeros(1, 10, 64, 64),
            target_numeric=torch.zeros(1, 4, 9),
            target_brush_ids=torch.zeros(1, 4, dtype=torch.long),
            target_present=torch.zeros(1, 4),
            target_padding_mask=torch.ones(1, 4, dtype=torch.bool),
            sample_ids=("sample_a",),
            patch_bounds=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            changed=torch.tensor([False]),
        )
        high_present = VisualDeltaPredictionOutput(
            pred_numeric=torch.zeros(1, 4, 9),
            pred_brush_logits=torch.zeros(1, 4, 8),
            pred_present_logits=torch.full((1, 4), 4.0),
        )
        low_present = VisualDeltaPredictionOutput(
            pred_numeric=torch.zeros(1, 4, 9),
            pred_brush_logits=torch.zeros(1, 4, 8),
            pred_present_logits=torch.full((1, 4), -4.0),
        )

        self.assertGreater(
            float(compute_visual_delta_loss(high_present, batch, slot_aware_targets=True).present),
            float(compute_visual_delta_loss(low_present, batch, slot_aware_targets=True).present),
        )

    def test_anti_dot_and_color_losses_penalize_tiny_bright_marks(self) -> None:
        import torch

        from Source.Model import compute_anti_dot_loss, compute_color_clamp_loss

        patch_tensor = torch.zeros(1, 10, 64, 64)
        patch_tensor[:, 3:6] = 0.20
        patch_tensor[:, 9:10] = 1.0
        tiny_bright = torch.tensor([[[0.5, 0.5, 0.2, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0]]])
        target_like = torch.tensor([[[0.5, 0.5, 0.2, 0.20, 0.10, 1.0, 0.22, 0.20, 0.18]]])
        logits = torch.full((1, 1), 6.0)

        tiny_loss = compute_anti_dot_loss(tiny_bright, logits, patch_tensor) + compute_color_clamp_loss(
            tiny_bright,
            logits,
            patch_tensor,
        )
        target_like_loss = compute_anti_dot_loss(target_like, logits, patch_tensor) + compute_color_clamp_loss(
            target_like,
            logits,
            patch_tensor,
        )

        self.assertGreater(float(tiny_loss), float(target_like_loss))

    def test_size_distribution_loss_is_lower_for_target_like_prediction(self) -> None:
        import torch

        from Source.Model import compute_assigned_size_distribution_loss

        target = torch.tensor(
            [
                [
                    [0.2, 0.2, 0.1, 0.04, 0.02, 1.0, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.8, 0.20, 0.10, 1.0, 0.1, 0.1, 0.1],
                ]
            ]
        )
        collapsed = target.clone()
        collapsed[..., 2:5] = torch.tensor([0.2, 0.02, 0.01])
        mask = torch.ones(1, 2, dtype=torch.bool)

        self.assertLess(
            float(compute_assigned_size_distribution_loss(target, target, mask)),
            float(compute_assigned_size_distribution_loss(collapsed, target, mask)),
        )

    def test_missing_target_strokes_increase_recall_loss(self) -> None:
        import torch

        from Source.Model import compute_present_recall_loss

        target_count = torch.tensor([2.0])
        missing = compute_present_recall_loss(torch.full((1, 4), -6.0), target_count)
        covered = compute_present_recall_loss(torch.full((1, 4), 6.0), target_count)

        self.assertGreater(float(missing), float(covered))

    def test_soft_renderer_shapes_gradients_and_preservation(self) -> None:
        import torch

        from Source.Model import VisualDeltaBatch, VisualDeltaPredictionOutput, compute_visual_patch_loss, render_soft_strokes

        draft = torch.zeros(1, 3, 64, 64)
        numeric = torch.tensor([[[0.8, 0.8, 0.0, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]]], requires_grad=True)
        logits = torch.full((1, 1), 6.0, requires_grad=True)
        rendered = render_soft_strokes(draft, numeric, logits)
        self.assertEqual(rendered.shape, (1, 3, 64, 64))
        rendered.sum().backward(retain_graph=True)
        self.assertIsNotNone(numeric.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(numeric.grad)))
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

        patch_tensor = torch.zeros(1, 10, 64, 64)
        patch_tensor[:, 9:10, :16, :16] = 1.0
        batch = VisualDeltaBatch(
            patch_tensor=patch_tensor,
            target_numeric=torch.zeros(1, 1, 9),
            target_brush_ids=torch.zeros(1, 1, dtype=torch.long),
            target_present=torch.zeros(1, 1),
            target_padding_mask=torch.ones(1, 1, dtype=torch.bool),
            sample_ids=("sample_a",),
            patch_bounds=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            changed=torch.tensor([True]),
        )
        prediction = VisualDeltaPredictionOutput(
            pred_numeric=numeric,
            pred_brush_logits=torch.zeros(1, 1, 8),
            pred_present_logits=logits,
        )
        _, preservation, gradient, edge, low_frequency = compute_visual_patch_loss(prediction, batch)
        self.assertGreater(float(preservation.detach()), 0.0)
        self.assertTrue(torch.isfinite(gradient))
        self.assertTrue(torch.isfinite(edge))
        self.assertTrue(torch.isfinite(low_frequency))

    def test_paint_transformer_soft_renderer_shapes_gradients_and_present_logits(self) -> None:
        import torch

        from Source.Model import render_paint_transformer_soft_strokes

        draft = torch.zeros(1, 3, 64, 64)
        numeric = torch.tensor([[[0.5, 0.5, 0.25, 0.45, 0.08, 1.0, 0.9, 0.2, 0.1]]], requires_grad=True)
        high_logits = torch.full((1, 1), 6.0, requires_grad=True)
        low_logits = torch.full((1, 1), -6.0)

        high_render = render_paint_transformer_soft_strokes(draft, numeric, high_logits)
        low_render = render_paint_transformer_soft_strokes(draft, numeric.detach(), low_logits)
        high_render.sum().backward()

        self.assertEqual(high_render.shape, (1, 3, 64, 64))
        self.assertGreater(float(high_render.sum().detach()), float(low_render.sum().detach()))
        self.assertIsNotNone(numeric.grad)
        self.assertIsNotNone(high_logits.grad)
        self.assertTrue(torch.all(torch.isfinite(numeric.grad)))
        self.assertTrue(torch.all(torch.isfinite(high_logits.grad)))

    def test_structure_target_requirement_rejects_legacy_sample(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root)
            _write_manifest(split_root, ["sample_a"])

            with self.assertRaisesRegex(ValueError, "target_selection_mode"):
                VisualDeltaStrokeDataset(split_root, require_structure_targets=True)

    def test_structure_target_requirement_accepts_structure_sample(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, structure_target=True)
            _write_manifest(split_root, ["sample_a"])

            dataset = VisualDeltaStrokeDataset(
                split_root,
                patch_size=128,
                patch_stride=128,
                negative_patch_ratio=0.0,
                require_structure_targets=True,
            )

            self.assertGreater(len(dataset), 0)

    def test_target_contract_requirement_rejects_old_contract(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, target_contract="structure_first_v1")
            _write_manifest(split_root, ["sample_a"])

            with self.assertRaisesRegex(ValueError, "target_contract"):
                VisualDeltaStrokeDataset(
                    split_root,
                    require_target_contract="paint_transformer_output_detail_pair_v1",
                )

    def test_target_contract_requirement_accepts_output_detail_sample(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(
                split_root,
                target_contract="paint_transformer_output_detail_pair_v1",
            )
            _write_manifest(split_root, ["sample_a"])

            dataset = VisualDeltaStrokeDataset(
                split_root,
                patch_size=128,
                patch_stride=128,
                negative_patch_ratio=0.0,
                require_target_contract="paint_transformer_output_detail_pair_v1",
            )

            self.assertGreater(len(dataset), 0)

    def test_structure_first_selection_prefers_high_impact_strokes(self) -> None:
        from Source.Synthetic.prepare_train_val_test import (
            TARGET_SELECTION_STRUCTURE_FIRST,
            _select_dataset_strokes,
        )

        base = [_stroke(0.1, 0.1)]
        tiny_texture = _stroke(0.2, 0.2)
        tiny_texture["length"] = 0.002
        tiny_texture["width"] = 0.001
        big_shape = _stroke(0.8, 0.8)
        big_shape["length"] = 0.08
        big_shape["width"] = 0.04
        selected, base_count, finishing_count, adjusted, manifest = _select_dataset_strokes(
            source_dir=Path("sample"),
            strokes=base + [tiny_texture, big_shape],
            requested_base_count=1,
            requested_finishing_count=1,
            completion_ratio=0.5,
            target_selection_mode=TARGET_SELECTION_STRUCTURE_FIRST,
        )

        self.assertFalse(adjusted)
        self.assertEqual(base_count, 1)
        self.assertEqual(finishing_count, 1)
        self.assertAlmostEqual(selected[1]["x"], big_shape["x"])
        self.assertEqual(manifest["target_selection_mode"], "structure_first_v1")
        self.assertEqual(manifest["selected_source_indices"], [2])

    def test_sobel_gradient_loss_detects_shifted_edges(self) -> None:
        import torch

        from Source.Model import compute_gradient_loss

        target = torch.zeros(1, 3, 64, 64)
        target[:, :, :, 32:] = 1.0
        identical = target.clone()
        shifted = torch.zeros_like(target)
        shifted[:, :, :, 36:] = 1.0
        mask = torch.ones(1, 1, 64, 64)

        identical_loss = compute_gradient_loss(identical, target, mask)
        shifted_loss = compute_gradient_loss(shifted, target, mask)

        self.assertLess(float(identical_loss), 1e-5)
        self.assertGreater(float(shifted_loss), float(identical_loss))

    def test_edge_alignment_loss_is_finite_and_differentiable(self) -> None:
        import torch

        from Source.Model import compute_edge_alignment_loss

        draft = torch.zeros(1, 3, 64, 64)
        target = draft.clone()
        target[:, :, 16:48, 16:48] = 1.0
        predicted = torch.zeros_like(target, requires_grad=True)
        predicted.data[:, :, 20:52, 20:52] = 1.0
        mask = torch.ones(1, 1, 64, 64)

        loss = compute_edge_alignment_loss(draft, predicted, target, mask)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.all(torch.isfinite(predicted.grad)))

    def test_edge_focused_sampling_scores_structure_above_flat_patch(self) -> None:
        from Source.Model.visual_delta_dataset import _patch_priority_score

        edge_score = _patch_priority_score(
            target_stroke_count=1,
            edge_density=0.05,
            error_score=0.20,
            edge_focused_sampling=True,
        )
        flat_score = _patch_priority_score(
            target_stroke_count=1,
            edge_density=0.0,
            error_score=0.01,
            edge_focused_sampling=True,
        )

        self.assertGreater(edge_score, flat_score)

    def test_diagnostics_reject_mad_only_noisy_improvement(self) -> None:
        import numpy as np
        from PIL import Image

        from Source.Model.prediction_diagnostics import compute_prediction_diagnostics

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft = np.zeros((64, 64, 3), dtype=np.uint8)
            target = draft.copy()
            target[:, 16:] = 255
            predicted = np.full((64, 64, 3), 16, dtype=np.uint8)
            Image.fromarray(draft).save(root / "draft.png")
            Image.fromarray(target).save(root / "target.png")
            Image.fromarray(predicted).save(root / "predicted.png")

            diagnostics = compute_prediction_diagnostics(
                root / "draft.png",
                root / "target.png",
                root / "predicted.png",
                predicted_strokes=[_stroke(0.5, 0.5)],
                target_strokes=[_stroke(0.75, 0.5)],
                min_changed_pixel_ratio=0.0,
                min_edge_overlap=0.5,
            )

        self.assertEqual(diagnostics["status"], "failed_structure_noise")

    def test_export_candidate_filter_removes_tiny_dot(self) -> None:
        import torch

        from Source.Model.export_visual_delta_predictions import _visual_delta_candidate_score

        patch_tensor = torch.zeros(10, 64, 64)
        patch_tensor[3:6, 24:40, 24:40] = 0.5
        patch_tensor[6:9, 24:40, 24:40] = 0.5
        patch_tensor[6:9, 32, 32] = 1.0
        patch_tensor[9, 32, 32] = 1.0
        tiny = torch.tensor([0.5, 0.5, 0.0, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0])
        useful = torch.tensor([0.5, 0.5, 0.0, 0.20, 0.10, 1.0, 0.5, 0.5, 0.5])
        wrong_color = torch.tensor([0.5, 0.5, 0.0, 0.20, 0.10, 1.0, 1.0, 1.0, 1.0])

        tiny_score, tiny_area = _visual_delta_candidate_score(
            tiny,
            present_score=0.99,
            patch_tensor=patch_tensor,
            min_render_area=8.0,
        )
        useful_score, useful_area = _visual_delta_candidate_score(
            useful,
            present_score=0.99,
            patch_tensor=patch_tensor,
            min_render_area=8.0,
        )
        wrong_color_score, _ = _visual_delta_candidate_score(
            wrong_color,
            present_score=0.99,
            patch_tensor=patch_tensor,
            min_render_area=8.0,
        )

        self.assertLess(tiny_area, 8.0)
        self.assertEqual(tiny_score, 0.0)
        self.assertGreater(useful_area, 8.0)
        self.assertGreater(useful_score, 0.0)
        self.assertLess(wrong_color_score, useful_score)

    def test_export_candidate_color_uses_target_footprint(self) -> None:
        import torch

        from Source.Model.export_visual_delta_predictions import _candidate_numeric_with_residual_color

        patch_tensor = torch.zeros(10, 32, 32)
        patch_tensor[3, 12:20, 12:20] = 0.8
        patch_tensor[4, 12:20, 12:20] = 0.3
        patch_tensor[5, 12:20, 12:20] = 0.1
        patch_tensor[9, 12:20, 12:20] = 1.0
        numeric = torch.tensor([0.5, 0.5, 0.0, 0.25, 0.25, 1.0, 0.0, 0.0, 1.0])

        corrected = _candidate_numeric_with_residual_color(numeric, patch_tensor)

        self.assertGreater(float(corrected[6]), 0.7)
        self.assertLess(float(corrected[7]), 0.4)
        self.assertLess(float(corrected[8]), 0.2)

    def test_export_candidate_selection_respects_sample_cap(self) -> None:
        from Source.Model.export_visual_delta_predictions import _select_ranked_export_candidates

        candidates = [
            {"stroke": {"x": index}, "score": float(index), "present_score": 0.9, "order": index}
            for index in range(10)
        ]

        selected = _select_ranked_export_candidates(candidates, max_strokes_per_sample=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual([entry["stroke"]["x"] for entry in selected], [7, 8, 9])

    def test_export_summary_rejects_low_change_improvement(self) -> None:
        from Source.Model.export_visual_delta_predictions import _export_summary

        summary = _export_summary(
            [
                {
                    "status": "failed_low_pixel_change",
                    "visual_improved": True,
                }
            ]
        )

        self.assertEqual(summary["visual_improvement_rate"], 1.0)
        self.assertEqual(summary["low_change_rate"], 1.0)
        self.assertEqual(summary["checkpoint_status"], "visual_failed")

    def test_visual_delta_export_rejects_non_best_checkpoint_by_default(self) -> None:
        from pathlib import Path

        from Source.Model.export_visual_delta_predictions import _validate_export_checkpoint

        checkpoint = {"checkpoint_type": "epoch"}

        with self.assertRaisesRegex(ValueError, "refusing to export"):
            _validate_export_checkpoint(checkpoint, Path("latest.pt"), allow_visual_failed=False)

        _validate_export_checkpoint(checkpoint, Path("latest.pt"), allow_visual_failed=True)

    def test_image_delta_compiler_emits_renderable_strokes(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_image_delta_strokes import compile_image_delta_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (128, 128), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((48, 48, 96, 96), fill=(220, 180, 120))
            draft.save(draft_path)
            target.save(target_path)

            strokes = compile_image_delta_strokes(
                draft_path,
                target_path,
                cell_size=32,
                stride=32,
                max_strokes=8,
                min_error=0.04,
                min_cell_changed_pixels=8,
                min_stroke_pixels=12,
            )

        self.assertGreater(len(strokes), 0)
        self.assertLessEqual(len(strokes), 8)
        for stroke in strokes:
            self.assertEqual(stroke["brush"], "paint_transformer_rect")
            self.assertGreaterEqual(stroke["length"], 12 / 128)
            self.assertGreaterEqual(stroke["width"], 12 / 128)
            self.assertTrue(all(0.0 <= channel <= 1.0 for channel in stroke["color"]))

    def test_image_delta_default_strokes_are_rectangular(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_image_delta_strokes import compile_image_delta_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (128, 128), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((32, 40, 112, 88), fill=(220, 180, 120))
            draft.save(draft_path)
            target.save(target_path)

            strokes = compile_image_delta_strokes(draft_path, target_path, max_strokes=16)

        self.assertGreater(len(strokes), 0)
        mean_ratio = sum(stroke["length"] / stroke["width"] for stroke in strokes) / len(strokes)
        self.assertGreaterEqual(mean_ratio, 1.25)

    def test_image_delta_source_target_uses_original_image(self) -> None:
        from PIL import Image

        from Source.Model.export_image_delta_strokes import _write_target_image

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            sample_dir = root / "sample"
            output_dir = root / "output"
            sample_dir.mkdir()
            Image.new("RGB", (8, 8), (20, 20, 20)).save(sample_dir / "draft.png")
            Image.new("RGB", (8, 8), (255, 0, 0)).save(sample_dir / "finished.png")
            source_path = root / "source.png"
            Image.new("RGB", (16, 16), (0, 0, 255)).save(source_path)

            target_path = _write_target_image(
                sample_dir,
                {
                    "draft_image": "draft.png",
                    "finished_image": "finished.png",
                    "source_image": str(source_path),
                },
                output_dir,
                "source-image",
            )

            with Image.open(target_path) as target_image:
                pixel = target_image.convert("RGB").getpixel((0, 0))

        self.assertEqual(pixel, (0, 0, 255))

    def test_image_delta_compiler_uses_local_orientation(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_image_delta_strokes import compile_image_delta_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (128, 128), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.line((24, 24, 104, 104), fill=(220, 180, 120), width=10)
            draft.save(draft_path)
            target.save(target_path)

            strokes = compile_image_delta_strokes(
                draft_path,
                target_path,
                cell_size=96,
                stride=96,
                max_strokes=1,
                min_error=0.04,
                min_cell_changed_pixels=8,
                min_stroke_pixels=8,
                aspect_ratio=2.0,
            )

        self.assertEqual(len(strokes), 1)
        stroke = strokes[0]
        self.assertGreater(stroke["length"], stroke["width"] * 1.5)
        self.assertTrue(0.05 <= stroke["angle"] <= 0.20)

    def test_image_delta_summary_rejects_low_change_improvement(self) -> None:
        from Source.Model.export_image_delta_strokes import _export_summary

        summary = _export_summary(
            [
                {
                    "status": "failed_low_pixel_change",
                    "visual_improved": True,
                }
            ]
        )

        self.assertEqual(summary["visual_improvement_rate"], 1.0)
        self.assertEqual(summary["low_change_rate"], 1.0)
        self.assertEqual(summary["checkpoint_status"], "visual_failed")

    def test_image_delta_sample_filter_selects_requested_ids(self) -> None:
        from Source.Model.export_image_delta_strokes import _select_manifest_samples

        selected = _select_manifest_samples(
            [{"path": "paint_transformer_000001"}, {"path": "paint_transformer_000005"}],
            ("paint_transformer_000005",),
            limit=1,
        )

        self.assertEqual(selected, [{"path": "paint_transformer_000005"}])

    def test_greedy_candidate_proposal_uses_varied_size_tiers(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_greedy_stroke_optimizer import propose_greedy_candidates

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (128, 128), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((24, 32, 104, 96), fill=(220, 180, 120))
            draft.save(draft_path)
            target.save(target_path)

            candidates = propose_greedy_candidates(
                draft_path,
                target_path,
                size_tiers=(64, 32, 16),
                max_component_anchors=1,
                max_point_anchors=0,
            )

        unique_lengths = {round(candidate["stroke"]["length"], 4) for candidate in candidates}
        self.assertGreaterEqual(len(unique_lengths), 3)

    def test_greedy_optimizer_accepts_biggest_improving_tier_first(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_greedy_stroke_optimizer import optimize_greedy_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (128, 128), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((24, 44, 104, 84), fill=(220, 180, 120))
            draft.save(draft_path)
            target.save(target_path)

            strokes, manifest, _ = optimize_greedy_strokes(
                draft_path,
                target_path,
                size_tiers=(64, 24),
                angle_degrees=(0.0,),
                opacities=(0.85,),
                max_strokes=1,
                min_stroke_mad_improvement=0.0,
                target_mad_threshold=0.0,
                max_component_anchors=1,
                max_point_anchors=0,
                aspect_ratio=2.0,
            )

        self.assertEqual(len(strokes), 1)
        self.assertAlmostEqual(strokes[0]["length"], 64 / 128)
        self.assertEqual(manifest["history"][0]["size_pixels"], 64)
        self.assertGreater(manifest["estimated_mad_improvement"], 0.0)

    def test_greedy_detail_cadence_prioritizes_small_tiers(self) -> None:
        from Source.Model.export_greedy_stroke_optimizer import _tier_search_groups

        first_groups = _tier_search_groups(
            stroke_index=1,
            size_tiers=(64, 32, 16),
            detail_size_tiers=(16,),
            detail_start_stroke=2,
            detail_cadence=1,
            coarse_min_improvement=0.03,
            detail_min_improvement=0.006,
        )
        detail_groups = _tier_search_groups(
            stroke_index=2,
            size_tiers=(64, 32, 16),
            detail_size_tiers=(16,),
            detail_start_stroke=2,
            detail_cadence=1,
            coarse_min_improvement=0.03,
            detail_min_improvement=0.006,
        )

        self.assertEqual(first_groups[0]["phase"], "coarse")
        self.assertEqual(detail_groups[0]["phase"], "detail")
        self.assertEqual(detail_groups[0]["size_tiers"], (16,))
        self.assertEqual(detail_groups[0]["min_improvement"], 0.006)
        self.assertEqual(detail_groups[1]["phase"], "coarse")

    def test_greedy_anchor_border_margin_trims_edge_errors(self) -> None:
        import numpy as np

        from Source.Model.export_greedy_stroke_optimizer import _apply_anchor_border_margin

        mask = np.ones((8, 8), dtype=bool)
        trimmed = _apply_anchor_border_margin(mask, 2)

        self.assertFalse(bool(trimmed[0, 4]))
        self.assertFalse(bool(trimmed[4, 0]))
        self.assertFalse(bool(trimmed[7, 4]))
        self.assertFalse(bool(trimmed[4, 7]))
        self.assertTrue(bool(trimmed[4, 4]))

    def test_greedy_optimizer_rejects_non_improving_images(self) -> None:
        from PIL import Image

        from Source.Model.export_greedy_stroke_optimizer import optimize_greedy_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            Image.new("RGB", (64, 64), (80, 80, 80)).save(draft_path)
            Image.new("RGB", (64, 64), (80, 80, 80)).save(target_path)

            strokes, manifest, _ = optimize_greedy_strokes(
                draft_path,
                target_path,
                max_strokes=8,
            )

        self.assertEqual(strokes, [])
        self.assertEqual(manifest["accepted_stroke_count"], 0)
        self.assertEqual(manifest["stop_reason"], "target_threshold")

    def test_greedy_force_max_strokes_bypasses_improvement_threshold(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_greedy_stroke_optimizer import optimize_greedy_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (64, 64), (32, 32, 32))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((20, 20, 44, 44), fill=(220, 180, 120))
            draft.save(draft_path)
            target.save(target_path)

            rejected, rejected_manifest, _ = optimize_greedy_strokes(
                draft_path,
                target_path,
                size_tiers=(16,),
                angle_degrees=(0.0,),
                opacities=(0.85,),
                max_strokes=1,
                min_stroke_mad_improvement=999.0,
                target_mad_threshold=0.0,
                max_component_anchors=1,
                max_point_anchors=0,
            )
            forced, forced_manifest, _ = optimize_greedy_strokes(
                draft_path,
                target_path,
                size_tiers=(16,),
                angle_degrees=(0.0,),
                opacities=(0.85,),
                max_strokes=1,
                min_stroke_mad_improvement=999.0,
                target_mad_threshold=0.0,
                max_component_anchors=1,
                max_point_anchors=0,
                force_max_strokes=True,
            )

        self.assertEqual(rejected, [])
        self.assertEqual(rejected_manifest["stop_reason"], "no_improving_candidate")
        self.assertEqual(len(forced), 1)
        self.assertTrue(forced_manifest["history"][0]["forced_accept"])
        self.assertEqual(forced_manifest["stop_reason"], "max_strokes")

    def test_greedy_residual_color_estimation_improves_patch(self) -> None:
        import numpy as np

        from Source.Model.export_greedy_stroke_optimizer import _composite_color, _estimate_residual_color

        current = np.zeros((8, 8, 3), dtype=np.float32)
        target = np.zeros((8, 8, 3), dtype=np.float32)
        target[..., 0] = 1.0
        alpha = np.ones((8, 8), dtype=np.float32) * 0.5

        color = _estimate_residual_color(current, target, alpha)
        rendered = _composite_color(current, color, alpha)

        self.assertGreater(color[0], 0.9)
        self.assertLess(float(np.abs(target - rendered).mean()), float(np.abs(target - current).mean()))

    def test_greedy_source_target_uses_original_image(self) -> None:
        from PIL import Image

        from Source.Model.export_greedy_stroke_optimizer import GreedyStrokeOptimizerConfig, _export_sample

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            sample_dir = root / "sample"
            output_dir = root / "output"
            sample_dir.mkdir()
            Image.new("RGB", (8, 8), (20, 20, 20)).save(sample_dir / "draft.png")
            Image.new("RGB", (8, 8), (255, 0, 0)).save(sample_dir / "finished.png")
            source_path = root / "source.png"
            Image.new("RGB", (16, 16), (0, 0, 255)).save(source_path)
            _write_json(sample_dir / "base_strokes.json", _program([_stroke(0.1, 0.1)]))
            _write_json(sample_dir / "finishing_strokes.json", _program([_stroke(0.2, 0.2)]))
            sample = {
                "version": 1,
                "sample_id": "sample_a",
                "source_image": str(source_path),
                "base_strokes": "base_strokes.json",
                "finishing_strokes": "finishing_strokes.json",
                "draft_image": "draft.png",
                "finished_image": "finished.png",
            }

            _export_sample(
                sample_dir,
                sample,
                output_dir,
                GreedyStrokeOptimizerConfig(render=False, max_strokes=1, target_mode="source-image"),
            )

            with Image.open(output_dir / "target.png") as target_image:
                pixel = target_image.convert("RGB").getpixel((0, 0))

        self.assertEqual(pixel, (0, 0, 255))

    def test_target_retrieval_summary_rejects_low_change_improvement(self) -> None:
        from Source.Model.export_target_stroke_retrieval import _export_summary

        summary = _export_summary(
            [
                {
                    "status": "failed_low_pixel_change",
                    "visual_improved": True,
                }
            ]
        )

        self.assertEqual(summary["visual_improvement_rate"], 1.0)
        self.assertEqual(summary["low_change_rate"], 1.0)
        self.assertEqual(summary["checkpoint_status"], "visual_failed")

    def test_recursive_duplicate_key_quantizes_near_identical_strokes(self) -> None:
        from Source.Model.export_visual_delta_predictions import _stroke_key

        first = _stroke(0.123456, 0.654321)
        second = _stroke(0.123459, 0.654319)

        self.assertEqual(_stroke_key(first), _stroke_key(second))

    def test_runtime_patch_tensor_uses_current_draft(self) -> None:
        import torch

        from Source.Model.export_visual_delta_predictions import _runtime_patch_tensor

        first_draft = torch.zeros(3, 512, 512)
        second_draft = torch.ones(3, 512, 512) * 0.25
        target = torch.ones(3, 512, 512)
        first_error = torch.abs(target - first_draft)
        second_error = torch.abs(target - second_draft)
        first_mask = torch.ones(1, 512, 512)
        second_mask = torch.ones(1, 512, 512)

        first_patch = _runtime_patch_tensor(first_draft, target, first_error, first_mask, left=0, top=0, patch_size=64)
        second_patch = _runtime_patch_tensor(second_draft, target, second_error, second_mask, left=0, top=0, patch_size=64)

        self.assertFalse(torch.equal(first_patch[0:3], second_patch[0:3]))
        self.assertLess(float(second_patch[6:9].mean()), float(first_patch[6:9].mean()))

    def test_strokes_per_pass_caps_recursive_selection(self) -> None:
        from Source.Model.export_visual_delta_predictions import _select_ranked_export_candidates

        candidates = [
            {"stroke": {"x": index}, "score": 100.0 - index, "present_score": 0.9, "order": index}
            for index in range(600)
        ]

        selected = _select_ranked_export_candidates(candidates, max_strokes_per_sample=256)

        self.assertEqual(len(selected), 256)

    def test_target_stroke_retrieval_scores_error_region_higher(self) -> None:
        import numpy as np

        from Source.Model.export_target_stroke_retrieval import score_target_stroke

        error = np.zeros((512, 512), dtype=np.float32)
        error[360:430, 360:430] = 1.0
        low_error = _stroke(0.1, 0.1)
        high_error = _stroke(0.78, 0.78)

        self.assertGreater(score_target_stroke(high_error, error), score_target_stroke(low_error, error))

    def test_target_stroke_retrieval_ranks_from_image_error(self) -> None:
        from PIL import Image, ImageDraw

        from Source.Model.export_target_stroke_retrieval import rank_target_strokes

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            draft_path = root / "draft.png"
            target_path = root / "target.png"
            draft = Image.new("RGB", (512, 512), (0, 0, 0))
            target = draft.copy()
            draw = ImageDraw.Draw(target)
            draw.rectangle((360, 360, 430, 430), fill=(255, 255, 255))
            draft.save(draft_path)
            target.save(target_path)
            strokes = [_stroke(0.1, 0.1), _stroke(0.78, 0.78)]

            ranked = rank_target_strokes(strokes, draft_path, target_path)

            self.assertEqual(ranked[0]["source_index"], 1)

    def test_tiny_training_smoke_writes_checkpoint(self) -> None:
        import torch

        from Source.Model.train_visual_delta_strokes import VisualDeltaTrainingConfig, train_visual_delta_strokes

        with tempfile.TemporaryDirectory() as root_name:
            data_root = Path(root_name) / "Data"
            output_dir = Path(root_name) / "Checkpoints"
            _write_data_root(data_root)

            result = train_visual_delta_strokes(
                VisualDeltaTrainingConfig(
                    data_root=data_root,
                    output_dir=output_dir,
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    num_workers=0,
                    model_dim=16,
                    hidden_dim=8,
                    decoder_layers=1,
                    num_heads=4,
                    ff_dim=32,
                    dropout=0.0,
                    patch_size=128,
                    patch_stride=128,
                    grid_size=2,
                    max_strokes_per_patch=4,
                    edge_focused_sampling=True,
                    overfit_patches=1,
                    visual_validation_samples=0,
                    require_target_contract=None,
                )
            )
            checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")

            self.assertEqual(result["train_patch_count"], 1)
            self.assertIn("model_config", checkpoint)
            self.assertIn("dataset_config", checkpoint)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertFalse((output_dir / "best.pt").exists())

    def test_overfit_samples_selects_whole_sample_patch_set(self) -> None:
        from Source.Model import VisualDeltaStrokeDataset
        from Source.Model.train_visual_delta_strokes import _first_sample_indices

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, finishing_strokes=[_stroke(0.2, 0.2)])
            _write_manifest(split_root, ["sample_a"])

            dataset = VisualDeltaStrokeDataset(split_root, patch_size=128, patch_stride=128, negative_patch_ratio=0.0)
            indices = _first_sample_indices(dataset, 1)
            selected_target_counts = [dataset.patch_index[index].target_stroke_count for index in indices]

            self.assertGreater(len(indices), 1)
            self.assertIn(0, selected_target_counts)
            self.assertIn(1, selected_target_counts)


if __name__ == "__main__":
    unittest.main()
