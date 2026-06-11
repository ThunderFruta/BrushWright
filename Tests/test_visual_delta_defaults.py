from __future__ import annotations

import unittest


class VisualDeltaDefaultsTest(unittest.TestCase):
    def test_prepare_split_defaults_use_coarse_detail_split_ratio(self) -> None:
        from Source.PaintTransformerReference import synthesize_samples
        from Source.Synthetic import prepare_train_val_test

        self.assertEqual(synthesize_samples.DEFAULT_OUTPUT_ROOT.parts[:2], ("Outputs", "Latest"))
        self.assertEqual(prepare_train_val_test.DEFAULT_SOURCE_ROOT.parts[:3], ("Outputs", "Latest", "PaintTransformerSamples"))
        self.assertAlmostEqual(prepare_train_val_test.DEFAULT_COMPLETION_RATIO, 0.5)
        self.assertAlmostEqual(prepare_train_val_test.DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO, 3.0 / 5.0)
        self.assertEqual(prepare_train_val_test.DEFAULT_DRAFT_IMAGE_MIN_COMPLETION, 0.50)
        self.assertEqual(prepare_train_val_test.DEFAULT_DRAFT_IMAGE_MAX_COMPLETION, 0.70)
        self.assertAlmostEqual(
            synthesize_samples.DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
            prepare_train_val_test.DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
        )

    def test_source_generation_default_selects_coarse_native_frame(self) -> None:
        from Source.PaintTransformerReference.synthesize_samples import (
            DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
            _native_frame_index_for_ratio,
        )

        self.assertEqual(_native_frame_index_for_ratio(6, DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO), 3)

    def test_visual_delta_training_defaults_use_dense_proposals(self) -> None:
        from Source.Model.train_visual_delta_strokes import VisualDeltaTrainingConfig

        config = VisualDeltaTrainingConfig()

        self.assertEqual(config.model_dim, 768)
        self.assertEqual(config.hidden_dim, 192)
        self.assertEqual(config.decoder_layers, 10)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.ff_dim, 3072)
        self.assertEqual(config.patch_size, 128)
        self.assertEqual(config.patch_stride, 64)
        self.assertEqual(config.grid_size, 16)
        self.assertEqual(config.max_strokes_per_patch, 512)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.target_vram_gb, 48)
        self.assertEqual(config.training_renderer, "paint-transformer-soft")
        self.assertEqual(config.output_dir.name, "VisualDeltaStrokeCompilerV8UsableV1Large")
        self.assertEqual(config.require_target_contract, "paint_transformer_original_image_target_v1")
        self.assertTrue(config.slot_aware_targets)
        self.assertGreater(config.anti_dot_weight, 0.0)
        self.assertEqual(config.max_export_strokes_per_sample, 512)
        self.assertEqual(config.max_export_strokes_per_patch, 512)
        self.assertAlmostEqual(config.present_threshold, 0.05)
        self.assertGreater(config.present_positive_weight, 1.0)

    def test_visual_delta_model_defaults_include_mixed_scale_proposals(self) -> None:
        from Source.Model.visual_delta_predictor import VisualDeltaStrokeCompilerConfig

        config = VisualDeltaStrokeCompilerConfig()

        self.assertEqual(config.model_dim, 768)
        self.assertEqual(config.hidden_dim, 192)
        self.assertEqual(config.num_layers, 10)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.ff_dim, 3072)
        self.assertEqual(config.grid_size, 16)
        self.assertEqual(config.max_strokes, 512)
        self.assertGreater(config.coarse_grid_size, 0)
        self.assertGreaterEqual(
            config.coarse_grid_size * config.coarse_grid_size + config.detail_grid_rows * config.detail_grid_cols,
            config.max_strokes,
        )
        self.assertGreaterEqual(config.detail_max_length * config.detail_max_width * 128 * 128, 8.0)
        self.assertLess(config.detail_min_width, config.detail_max_width)
        self.assertLess(config.detail_min_length, config.detail_max_length)
        self.assertGreaterEqual(config.coarse_max_length * config.coarse_max_width * 128 * 128, 8.0)

    def test_visual_delta_training_uses_mixed_scale_slot_layout(self) -> None:
        from Source.Model.train_visual_delta_strokes import _slot_layout_for_max_strokes

        self.assertEqual(
            _slot_layout_for_max_strokes(256),
            {"coarse_grid_size": 8, "detail_grid_rows": 16, "detail_grid_cols": 16},
        )
        self.assertEqual(
            _slot_layout_for_max_strokes(512),
            {"coarse_grid_size": 11, "detail_grid_rows": 16, "detail_grid_cols": 32},
        )
        self.assertEqual(
            _slot_layout_for_max_strokes(128),
            {"coarse_grid_size": 5, "detail_grid_rows": 8, "detail_grid_cols": 16},
        )

    def test_visual_delta_export_defaults_are_single_pass_compatible(self) -> None:
        from Source.Model.export_visual_delta_predictions import ExportVisualDeltaConfig

        config = ExportVisualDeltaConfig()

        self.assertEqual(config.output_root.parts[:2], ("Outputs", "Latest"))
        self.assertEqual(config.output_root.name, "VisualDeltaPredictionsV8UsableV1Large")
        self.assertEqual(config.checkpoint.parent.name, "VisualDeltaStrokeCompilerV8UsableV1Large")
        self.assertIsNone(config.sample_id)
        self.assertAlmostEqual(config.present_threshold, 0.05)
        self.assertEqual(config.max_strokes_per_patch, 512)
        self.assertFalse(config.allow_visual_failed_checkpoint)
        self.assertEqual(config.recursive_passes, 1)
        self.assertEqual(config.strokes_per_pass, 512)
        self.assertFalse(config.stop_on_non_improvement)
        self.assertTrue(config.keep_structure_failed_passes)

    def test_prediction_export_defaults_use_latest_output_roots(self) -> None:
        from Source.Model.export_greedy_stroke_optimizer import GreedyStrokeOptimizerConfig
        from Source.Model.export_image_delta_strokes import ImageDeltaStrokeConfig
        from Source.Model.export_target_stroke_retrieval import TargetStrokeRetrievalConfig
        from Source.Model.export_test_predictions import ExportPredictionsConfig

        self.assertEqual(GreedyStrokeOptimizerConfig().output_root.parts[:2], ("Outputs", "Latest"))
        self.assertEqual(ImageDeltaStrokeConfig().output_root.parts[:2], ("Outputs", "Latest"))
        self.assertEqual(TargetStrokeRetrievalConfig().output_root.parts[:2], ("Outputs", "Latest"))
        self.assertEqual(ExportPredictionsConfig().output_root.parts[:2], ("Outputs", "Latest"))

    def test_image_delta_defaults_use_dense_renderable_cells(self) -> None:
        from Source.Model.export_image_delta_strokes import ImageDeltaStrokeConfig

        config = ImageDeltaStrokeConfig()

        self.assertEqual(config.cell_size, 20)
        self.assertEqual(config.stride, 14)
        self.assertEqual(config.max_strokes, 512)
        self.assertAlmostEqual(config.min_error, 0.025)
        self.assertEqual(config.min_cell_changed_pixels, 4)
        self.assertEqual(config.min_stroke_pixels, 4)
        self.assertAlmostEqual(config.stroke_scale, 0.9)
        self.assertAlmostEqual(config.aspect_ratio, 1.4)
        self.assertAlmostEqual(config.opacity, 0.70)
        self.assertEqual(config.target_mode, "target-image")
        self.assertEqual(config.recursive_passes, 1)
        self.assertTrue(config.stop_on_non_improvement)
        self.assertAlmostEqual(config.min_pass_mad_improvement, 0.10)
        self.assertAlmostEqual(config.target_mad_threshold, 3.0)

    def test_greedy_optimizer_defaults_use_source_target_and_size_tiers(self) -> None:
        from Source.Model.export_greedy_stroke_optimizer import GreedyStrokeOptimizerConfig

        config = GreedyStrokeOptimizerConfig()

        self.assertEqual(config.output_root.parts[-1], "GreedyStrokeOptimizerV1")
        self.assertEqual(config.target_mode, "source-image")
        self.assertEqual(config.max_strokes, 256)
        self.assertEqual(config.min_stroke_mad_improvement, 0.03)
        self.assertEqual(config.detail_min_stroke_mad_improvement, 0.006)
        self.assertEqual(config.target_mad_threshold, 3.0)
        self.assertGreater(config.size_tiers[0], config.size_tiers[-1])
        self.assertEqual(config.size_tiers, tuple(sorted(config.size_tiers, reverse=True)))
        self.assertEqual(config.detail_size_tiers, (28, 18, 10, 6))
        self.assertEqual(config.detail_start_stroke, 8)
        self.assertEqual(config.detail_cadence, 2)
        self.assertFalse(config.force_max_strokes)
        self.assertEqual(config.anchor_border_margin, 0)


if __name__ == "__main__":
    unittest.main()
