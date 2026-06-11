# Notices

## Paint Transformer

- Project: unofficial PyTorch reimplementation of `Paint Transformer: Feed Forward Neural Painting with Stroke Prediction`
- Upstream: `https://github.com/Huage001/PaintTransformer`
- Local source subset: `Source/PaintTransformerReference/`
- Local notice and license: `ThirdParty/PaintTransformer/`
- License: Apache-2.0
- Intended use: optional synthetic data teacher that converts source images into ordered stroke programs, which BrushWright then splits into base and finishing strokes

Only inference-time model and rendering helpers are imported. Training code, demos, sample images, and pretrained checkpoints are intentionally excluded. Keep checkpoints as external local artifacts and pass them explicitly to the sample builder.

## Project Dependency Policy

Prefer permissively licensed renderer dependencies for V1. Avoid adding GPL application code or renderer code unless the licensing impact is documented and explicitly accepted.

## Art Institute of Chicago

- Project: Art Institute of Chicago Open Access
- Upstream: `https://www.artic.edu/open-access`
- API: `https://api.artic.edu/`
- Optional importer: `Source/Synthetic/import_art_institute_chicago.py`
- License note: public-domain collection images and metadata are offered under Creative Commons Zero (CC0)
- Intended use: preferred artwork source-image corpus for Paint Transformer-backed supervised BrushWright sample generation

## Openclipart

- Project: Openclipart
- Upstream: `https://openclipart.org/`
- Optional importer: `Source/Synthetic/import_openclipart.py`
- License note: Openclipart states uploaded clipart uses Creative Commons Zero 1.0 Public Domain terms
- Intended use: preferred external source-image corpus for Paint Transformer-backed supervised BrushWright sample generation

## ArtBench

- Project: `The ArtBench Dataset: Benchmarking Generative Models with Artworks`
- Upstream: `https://github.com/liaopeiyuan/artbench`
- Optional importer: `Source/Synthetic/import_artbench.py`
- Default access path: Hugging Face dataset mirror `zguo0525/ArtBench`
- License note: distributed for fair-use research; image data is not MIT-compatible and imported images must remain external local artifacts ignored by Git
- Intended use: optional source-image corpus for Paint Transformer-backed supervised BrushWright sample generation
