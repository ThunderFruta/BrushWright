# Image Corpus

Place local source images here when generating Paint Transformer-backed synthetic samples.

These files can be large or private and are ignored by Git. The sample builder reads images from this folder, converts them into ordered Paint Transformer stroke programs, then writes reproducible BrushWright artifacts under `Outputs/PaintTransformerSamples/`.

Example:

```bash
python3 -m Source.PaintTransformerReference.synthesize_samples \
  --model-path /path/to/paint_transformer_model.pth \
  --image-dir Assets/ImageCorpus
```

## Art Institute of Chicago

Art Institute of Chicago is the preferred artwork corpus path for BrushWright because public-domain Open Access images and data are CC0 and the API is easy to filter. Import the default 5,000-image local subset with:

```bash
python3 -m Source.Synthetic.import_art_institute_chicago --limit 5000 --image-size 512
```

This writes JPEG images under `Assets/ImageCorpus/ArtInstituteChicago/` and a manifest under `Outputs/ArtInstituteChicago/`. Keep those originals intact, then remove near-uniform scan/page borders into the cropped corpus with:

```bash
python3 -m Source.Synthetic.preprocess_image_corpus --clear-existing
```

The cropped 512x512 source images are written under `Assets/ImageCorpus/ArtInstituteChicagoCropped/`, with a preprocessing manifest under `Outputs/ArtInstituteChicago/`. Use the cropped folder for Paint Transformer generation when borders are hurting sample quality.

## Openclipart

Openclipart is the preferred broad image corpus path for BrushWright because its uploaded clipart uses CC0/public-domain terms. Import a small local subset with:

```bash
python3 -m Source.Synthetic.import_openclipart --limit 64 --image-size 512
```

This writes PNG images under `Assets/ImageCorpus/Openclipart/` and a manifest under `Outputs/Openclipart/`.


## Google Quick, Draw!

Google Quick, Draw! is an optional line-art/doodle corpus. Import 200 rendered PNG images per class with:

```bash
python3 -m Source.Synthetic.import_google_quickdraw --per-class 200 --clear-existing
```

This writes PNG images under `Assets/ImageCorpus/GoogleQuickDraw/` and a CC-BY-4.0 attribution manifest under `Outputs/GoogleQuickDraw/`.

## ArtBench

ArtBench is an optional research-only art corpus path for broader painterly sources. Import a small local subset with:

```bash
python3 -m Source.Synthetic.import_artbench --limit 64 --image-size 512
```

This writes images under `Assets/ImageCorpus/ArtBench/` and a manifest under `Outputs/ArtBench/`. Imported ArtBench files are ignored by Git and should remain external because ArtBench is distributed for fair-use research rather than as MIT-compatible or project-owned data.
