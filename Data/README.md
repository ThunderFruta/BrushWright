# Training Data

`Data/Train/`, `Data/Val/`, and `Data/Test/` are local generated datasets for supervised BrushWright training.

Each sample directory contains:

```text
sample.json
full_program.json
base_strokes.json
finishing_strokes.json
draft.png
finished.png
split_manifest.json
draft_render/render_manifest.json
finished_render/render_manifest.json
```

The trusted V5 PaintTransformer split is built from `Outputs/PaintTransformerSamples/ArtInstituteChicago/` with:

- `Train`: 800 samples
- `Val`: 100 samples
- `Test`: 100 samples
- base strokes: copied from each PaintTransformer source sample
- finishing strokes: copied from each PaintTransformer source sample's detail split
- draft image: copied from the source sample's own PaintTransformer `draft.png` at roughly three-fifths completion
- target image: copied from the source sample's own fine final PaintTransformer `finished.png`
- target contract: `paint_transformer_output_detail_pair_v1`

Some simple source images produce fewer PaintTransformer strokes than the preferred split needs. The split records
`stroke_count_adjusted: true` when it has to adapt counts.

The draft is intentionally the same recognizable PaintTransformer draft visible under `Outputs/PaintTransformerSamples`.
The training task is fine detail addition over that base, not inpainting and not replaying exported strokes on a blank canvas.

Regenerate the source PaintTransformer samples first so each `full_program.json` contains at least `3072` selected strokes:

```bash
/usr/bin/python3 run.py --image-dir Assets/ImageCorpus/ArtInstituteChicago --output-root Outputs/PaintTransformerSamples/ArtInstituteChicago --yes
```

Then regenerate the split with:

```bash
python3 -m Source.Synthetic.prepare_train_val_test --clear-existing --use-output-detail-pair
```

The visual-delta model uses `64x64` patches by default so each patch usually contains no more than the one-shot
`64` stroke output limit. The sample folders are ignored by Git because they are generated data.
`Data/dataset_manifest.json` and each split's `dataset_manifest.json` describe the local build.
