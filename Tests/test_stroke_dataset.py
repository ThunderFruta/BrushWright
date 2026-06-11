from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
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


def _program(strokes: list[dict], split: str) -> dict:
    return {
        "version": 1,
        "canvas": {"width": 512, "height": 512},
        "metadata": {"split": split},
        "strokes": strokes,
    }


def _write_sample(
    split_root: Path,
    sample_id: str,
    base_count: int,
    finishing_count: int,
    adjusted: bool,
    v1_contract: bool = False,
) -> None:
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
    if v1_contract:
        sample.update(
            {
                "target_contract": "paint_transformer_resplit_v1",
                "render_draft_from_base": True,
                "draft_stroke_completion_delta": 0.0,
            }
        )
    _write_json(sample_dir / "sample.json", sample)
    _write_json(sample_dir / "base_strokes.json", _program([_stroke(index) for index in range(base_count)], "base"))
    _write_json(
        sample_dir / "finishing_strokes.json",
        _program([_stroke(index) for index in range(finishing_count)], "finishing"),
    )
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


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model dataset tests")
class StrokeDatasetTest(unittest.TestCase):
    def test_split_manifest_discovery_and_sample_count(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=3, finishing_count=128, adjusted=False)
            _write_sample(split_root, "sample_b", base_count=3, finishing_count=65, adjusted=True)
            _write_manifest(
                split_root,
                [
                    {"sample_id": "sample_a", "path": "sample_a"},
                    {"sample_id": "sample_b", "path": "sample_b"},
                ],
            )

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)

            self.assertEqual(dataset.manifest["sample_count"], 2)
            self.assertEqual(len(dataset), 4)
            self.assertEqual([chunk.sample_id for chunk in dataset.chunk_index], ["sample_a", "sample_a", "sample_b", "sample_b"])

    def test_exact_finishing_target_produces_sixteen_chunks(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_exact", base_count=4, finishing_count=1024, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_exact", "path": "sample_exact"}])

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)

            self.assertEqual(len(dataset), 16)
            self.assertEqual(dataset.chunk_index[0].chunk_start, 0)
            self.assertEqual(dataset.chunk_index[0].chunk_end, 64)
            self.assertEqual(dataset.chunk_index[-1].chunk_start, 960)
            self.assertEqual(dataset.chunk_index[-1].chunk_end, 1024)

    def test_adjusted_short_target_pads_final_chunk(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_short", base_count=4, finishing_count=70, adjusted=True)
            _write_manifest(split_root, [{"sample_id": "sample_short", "path": "sample_short"}])

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)
            item = dataset[1]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(item.chunk_start, 64)
            self.assertEqual(item.chunk_end, 70)
            self.assertTrue(item.stroke_count_adjusted)
            self.assertEqual(item.target_numeric.shape, (64, 9))
            self.assertEqual(item.target_padding_mask.tolist()[:6], [False, False, False, False, False, False])
            self.assertTrue(all(item.target_padding_mask.tolist()[6:]))

    def test_collate_returns_expected_shapes_and_masks(self) -> None:
        from Source.Model import BrushWrightStrokeDataset, collate_stroke_chunks

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=64, adjusted=False)
            _write_sample(split_root, "sample_b", base_count=2, finishing_count=5, adjusted=True)
            _write_manifest(
                split_root,
                [
                    {"sample_id": "sample_a", "path": "sample_a"},
                    {"sample_id": "sample_b", "path": "sample_b"},
                ],
            )

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)
            batch = collate_stroke_chunks([dataset[0], dataset[1]])

            self.assertEqual(batch.base_tokens.numeric.shape, (2, 4, 9))
            self.assertEqual(batch.base_tokens.padding_mask.tolist(), [[False, False, False, False], [False, False, True, True]])
            self.assertEqual(batch.target_numeric.shape, (2, 64, 9))
            self.assertEqual(batch.target_brush_ids.shape, (2, 64))
            self.assertEqual(batch.target_padding_mask.shape, (2, 64))
            self.assertEqual(batch.draft_images.shape, (2, 3, 512, 512))
            self.assertEqual(batch.goal_images.shape, (2, 3, 512, 512))
            self.assertEqual(batch.error_maps.shape, (2, 3, 512, 512))
            self.assertEqual(batch.sample_ids, ("sample_a", "sample_b"))
            self.assertEqual(batch.chunk_starts.tolist(), [0, 0])
            self.assertEqual(batch.chunk_ends.tolist(), [64, 5])
            self.assertEqual(batch.stroke_count_adjusted.tolist(), [False, True])

    def test_base_tokens_can_pass_through_stroke_encoder(self) -> None:
        from Source.Model import BrushWrightStrokeDataset, StrokeEncoder, StrokeEncoderConfig, collate_stroke_chunks

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=64, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_a", "path": "sample_a"}])

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)
            batch = collate_stroke_chunks([dataset[0]])
            encoder = StrokeEncoder(
                StrokeEncoderConfig(model_dim=32, num_layers=1, num_heads=4, ff_dim=64, dropout=0.0, max_strokes=4)
            )
            output = encoder(
                batch.base_tokens.numeric,
                batch.base_tokens.brush_ids,
                batch.base_tokens.padding_mask,
            )

            self.assertEqual(output.features.shape, (1, 4, 32))
            self.assertEqual(output.pooled.shape, (1, 32))

    def test_dataset_caches_repeated_sample_reads(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_a", base_count=4, finishing_count=128, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_a", "path": "sample_a"}])

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=4)
            self.assertEqual(dataset.cache_stats(), {"samples": 0, "targets": 0})

            first = dataset[0]
            second = dataset[1]

            self.assertEqual(first.sample_id, "sample_a")
            self.assertEqual(second.sample_id, "sample_a")
            self.assertEqual(dataset.cache_stats(), {"samples": 1, "targets": 2})

    def test_v1_contract_rejects_non_matching_sample(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(split_root, "sample_bad", base_count=4, finishing_count=64, adjusted=False)
            _write_manifest(split_root, [{"sample_id": "sample_bad", "path": "sample_bad"}])

            with self.assertRaisesRegex(ValueError, "V1 contract"):
                BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=192, require_v1_contract=True)

    def test_v1_contract_accepts_current_contract(self) -> None:
        from Source.Model import BrushWrightStrokeDataset

        with tempfile.TemporaryDirectory() as root_name:
            split_root = Path(root_name) / "Train"
            split_root.mkdir()
            _write_sample(
                split_root,
                "sample_v1",
                base_count=192,
                finishing_count=64,
                adjusted=False,
                v1_contract=True,
            )
            _write_manifest(split_root, [{"sample_id": "sample_v1", "path": "sample_v1"}])

            dataset = BrushWrightStrokeDataset(split_root, chunk_size=64, max_base_strokes=192, require_v1_contract=True)

            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
