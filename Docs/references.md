# References

BrushWright is MIT-licensed project code, with optional external data and third-party reference components kept behind explicit boundaries.

## Core Reference

- Paint Transformer: Feed Forward Neural Painting with Stroke Prediction
  - Upstream implementation used as reference: `https://github.com/Huage001/PaintTransformer`
  - Local subset: `Source/PaintTransformerReference/`
  - Local third-party notice: `ThirdParty/PaintTransformer/README.md`
  - License: Apache-2.0

## Optional Image Corpora

- Art Institute of Chicago Open Access
  - Open access page: `https://www.artic.edu/open-access`
  - API: `https://api.artic.edu/`
  - License note: public-domain collection images and metadata are offered under Creative Commons Zero.

- Openclipart
  - Site: `https://openclipart.org/`
  - License note: Openclipart states uploaded clipart uses Creative Commons Zero 1.0 Public Domain terms.

- Google Quick, Draw!
  - Dataset: `https://github.com/googlecreativelab/quickdraw-dataset`
  - License note: CC-BY-4.0; keep attribution metadata with imported/generated corpora.

- ArtBench
  - Project: `https://github.com/liaopeiyuan/artbench`
  - Dataset paper: The ArtBench Dataset: Benchmarking Generative Models with Artworks
  - License note: fair-use research data; imported images are external local artifacts and are not MIT-compatible project assets.

## Artifact Policy

Generated datasets, model checkpoints, downloaded image corpora, and Paint Transformer checkpoints are local artifacts. They are intentionally ignored by Git and are not covered by the BrushWright MIT project license unless an artifact explicitly says otherwise.
