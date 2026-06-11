# Structure and Data Flow

BrushWright is organized around stroke programs. The central contract is:

```text
rough draft image + optional base stroke history
        |
        v
predicted finishing strokes
        |
        v
renderer
        |
        v
finished image + replayable stroke plan
```

The current design direction is shifting from next-stroke prediction to visual delta stroke compilation. The system should learn to turn a target visual change into editable strokes. See `Docs/visual_delta_stroke_compiler.md` for the decision and retired pipeline notes.

The current implemented pipeline covers renderer and synthetic sample plumbing:

```text
stroke_program.json
        |
        v
Source/Renderer/stroke_schema.py
        |
        v
Source/Renderer/render_program.py
        |
        v
final.png + replay.gif + per-stroke PNG frames + render_manifest.json
```

The current synthetic sample path is template-based:

```text
Config/stroke_styles.json + seed
        |
        v
Source/Synthetic/Templates/<template>.py
        |
        v
base_strokes + finishing_strokes
        |
        +--> base_strokes.json                  -> renderer -> draft.png
        +--> finishing_strokes.json             -> supervised target
        +--> full_program.json (base + finish)  -> renderer -> finished.png
```

An optional image-conditioned reference path is also available:

```text
Assets/ImageCorpus/* + external Paint Transformer checkpoint
        |
        v
Source/PaintTransformerReference/
        |
        v
full_program.json (ordered image-derived strokes)
        |
        +--> base_strokes.json                  -> Paint Transformer renderer -> draft.png
        +--> finishing_strokes.json             -> supervised target
        +--> full_program.json (base + finish)  -> Paint Transformer renderer -> finished.png
```

Earlier model experiments trained on `base_strokes.json -> finishing_strokes.json` chunks, including an image/error-conditioned V2 path. Those experiments are retained for regression and comparison, but they are retired as the product direction because their numeric loss improved while rendered predictions failed visual improvement gates.

## Repository Structure

```text
BrushWright/
  App/              local inspection tools, comparison views, replay UI
  Assets/           real-world reference media and local manual test inputs
  Config/           explicit project configuration such as stroke style ranges
  Docs/             architecture notes, decisions, experiment notes
  Fixtures/         tiny committed examples for tests and smoke runs
  Model/            encoders, decoders, training, inference
  Outputs/          ignored generated samples and run outputs
  Scripts/          local build and run entrypoints
  Source/           implementation packages
    Metrics/        stroke, image, style, preservation, and report metrics
    Renderer/       stroke schema, renderer wrappers, replay/export logic
    Synthetic/      template generation, draft/finish splits, sample assembly
    PaintTransformerReference/ imported Paint Transformer model and renderer helpers
  Tests/            focused schema, renderer, synthetic data, and metrics tests
  ThirdParty/       external notices, licenses, and local checkpoint paths
```

Only some modules exist today. Missing modules should be added when their first real implementation lands, not as empty scaffolding.

`Assets/` is for real-world manual test media. `Fixtures/` is for tiny committed automated-test inputs. `Outputs/` is for generated synthetic samples, renderer smoke outputs, and local run artifacts; it is ignored by Git except for its README.

## Current Renderer Data Flow

1. A stroke program JSON file is passed to `Scripts/run_renderer.sh`.
2. `Scripts/run_renderer.sh` calls `python -m Source.Renderer.render_program`.
3. `Source/Renderer/render_program.py` loads and validates the program with `Source/Renderer/stroke_schema.py`.
4. The wrapper renders each stroke with Paint Transformer's brush-mask renderer and writes:

```text
output_dir/
  final.png
  replay.gif
  frames/
    frame_000001.png
    frame_000002.png
  render_manifest.json
```

`final.png` is the last per-stroke frame. `replay.gif` is built from the PNG frame sequence.

## Stroke Program Shape

V1 stroke programs are JSON files:

```json
{
  "version": 1,
  "canvas": {
    "width": 512,
    "height": 512
  },
  "metadata": {
    "sample_id": "example",
    "seed": 1
  },
  "strokes": [
    {
      "x": 0.5,
      "y": 0.5,
      "angle": 0.25,
      "length": 0.3,
      "width": 0.03,
      "color": [0.8, 0.2, 0.1],
      "opacity": 0.9,
      "brush": "flat_oil"
    }
  ]
}
```

Schema rules:

- `version` is currently `1`.
- `canvas.width` and `canvas.height` are positive integers, defaulting to `512`.
- `x`, `y`, `length`, and `width` are normalized floats.
- `x` and `y` are the stroke center.
- `angle` is normalized turns, where `1.0` equals one full rotation.
- `color` is RGB float `[0.0, 1.0]`.
- `opacity` is a normalized float.
- `brush` is a stable string identifier.

## Synthetic Sample Data Flow

The pre-ML dataset flow extends the renderer pipeline without changing its contract:

```text
Source/Synthetic/generate_programs.py
        |
        v
Source/Synthetic/Templates/
        |
        v
base_strokes.json + finishing_strokes.json
        |
        +--> base_strokes.json                  -> renderer -> draft.png
        |
        +--> finishing_strokes.json             -> supervised target
        |
        +--> full_program.json (base + finish)  -> renderer -> finished.png
```

Each generated sample should save enough metadata to reproduce it:

```text
sample_dir/
  sample.json
  full_program.json
  base_strokes.json
  finishing_strokes.json
  draft.png
  finished.png
  draft_render/render_manifest.json
  finished_render/render_manifest.json
```

The implemented sample builder writes render manifests under `draft_render/` and `finished_render/`, then copies their final images to `draft.png` and `finished.png`. `Source.Synthetic.split_strokes` remains a legacy/manual utility for full programs without template roles.

## Boundary Rules

- `Source/Synthetic/` may depend on `Source/Renderer/stroke_schema.py` for producing valid stroke programs.
- `Assets/` should contain real-world input media, not generated training datasets.
- `Outputs/` should contain generated samples and local run artifacts, not hand-authored source.
- `Source/Renderer/` owns stroke schema validation and the current Paint Transformer rendering entrypoint.
- `Model/` should consume dataset artifacts, not call synthetic generators directly during inference.
- `Source/Metrics/` should read rendered images and stroke JSON outputs, not mutate samples.
- `App/` should inspect existing artifacts and invoke CLI entrypoints; it should not own renderer logic.
- `ThirdParty/PaintTransformer/` holds only the upstream license/notice and local checkpoint path.

## Commands

Build a complete synthetic sample:

```bash
python3 -m Source.Synthetic.build_sample --seed 1
```

Default output:

```text
Outputs/Samples/sample_000001/
```

Build Paint Transformer-backed samples from a local image corpus:

```bash
python3 run.py
```

Or pass all values as flags:

```bash
python3 run.py \
  --model-path /path/to/paint_transformer_model.pth \
  --image-dir Assets/ImageCorpus \
  --output-root Outputs/PaintTransformerSamples \
  --workers 3 \
  --base-count 768 \
  --finishing-count 256 \
  --yes
```

By default, `draft.png` is a native Paint Transformer frame at roughly three-fifths completion, while `finished.png` is the fine final native output.

For images with fewer accepted Paint Transformer strokes than requested, sample generation records the requested counts in metadata and adapts the actual split downward. Re-run with `--skip-existing` to resume after an interruption or failed image without overwriting completed sample folders.

The lower-level module entrypoint is also available:

```bash
python3 -m Source.PaintTransformerReference.synthesize_samples \
  --model-path /path/to/paint_transformer_model.pth \
  --image-dir Assets/ImageCorpus
```

Default output:

```text
Outputs/PaintTransformerSamples/paint_transformer_000001/
```

Generate only a full stroke program:

```bash
python3 -m Source.Synthetic.generate_programs --seed 1 --output Outputs/Samples/sample_000001/full_program.json
```

Choose a specific template/style:

```bash
python3 -m Source.Synthetic.build_sample --seed 1 --template house_icon --style flat_vector
```

Split a full program manually:

```bash
python3 -m Source.Synthetic.split_strokes Outputs/Samples/sample_000001/full_program.json --output-dir Outputs/Samples/sample_000001
```

Render the committed fixture:

```bash
Scripts/run_renderer.sh Fixtures/sample_stroke_program.json Outputs/RenderSmoke
```


Import the preferred Art Institute of Chicago artwork corpus:

```bash
python3 -m Source.Synthetic.import_art_institute_chicago --limit 5000 --image-size 512
```

Remove near-uniform borders from imported Art Institute images before synthesis:

```bash
python3 -m Source.Synthetic.preprocess_image_corpus --clear-existing
```

Generate Paint Transformer-backed samples from the cropped Art Institute images:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/ArtInstituteChicagoCropped \
  --output-root Outputs/PaintTransformerSamples/ArtInstituteChicagoCropped \
  --limit 8 \
  --yes
```

Import the preferred Openclipart source-image corpus:

```bash
python3 -m Source.Synthetic.import_openclipart --limit 64 --image-size 512
```

Generate Paint Transformer-backed samples from imported Openclipart images:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/Openclipart \
  --output-root Outputs/PaintTransformerSamples/Openclipart \
  --limit 8 \
  --yes
```

Import an optional ArtBench source-image corpus:

```bash
python3 -m Source.Synthetic.import_artbench --limit 64 --image-size 512
```

Generate Paint Transformer-backed samples from imported ArtBench images:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/ArtBench \
  --output-root Outputs/PaintTransformerSamples/ArtBench \
  --limit 8 \
  --yes
```
