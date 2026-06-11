# BrushWright

BrushWright is a research-oriented painting assistant for turning rough or unfinished digital paintings into improved, style-matched finished work by generating editable brushstroke programs.

The core output is not just a finished image. BrushWright aims to produce the missing finishing strokes that can be replayed, inspected, edited, rendered, and evaluated.

## Project Thesis

Most AI painting tools generate pixels. BrushWright focuses on stroke-level completion:

```text
rough draft + optional base stroke history
        |
        v
predicted finishing strokes
        |
        v
renderer
        |
        v
finished painting + replayable stroke plan
```

For the first version, unfinishedness is defined objectively:

```text
finished_strokes = base_strokes + finishing_strokes
draft_image      = render(base_strokes)
target_output    = finishing_strokes
```

This avoids the subjective problem of deciding whether a real artwork is intentionally loose or genuinely incomplete.

## V1 Target

Train a model on synthetic 512x512 painterly examples:

- Generate a finished painting as an ordered stroke program.
- Split the program into base strokes and finishing strokes.
- Render the base strokes as the rough draft.
- Train the model to predict the withheld finishing strokes.
- Render the predicted strokes and compare against the hidden finished painting.

Initial constraints:

- Canvas: `512x512`
- Base strokes: `192`
- Finishing strokes: `64`
- Brush families: `2-3`
- Training target: supervised finishing-stroke prediction
- Renderer: Paint Transformer brush-mask renderer

## Expected Outputs

BrushWright should eventually produce:

- `added_strokes.json`
- rendered finished image
- before/after comparison
- stroke replay
- layer/stroke inspection
- evaluation report

Example stroke shape:

```json
{
  "x": 0.42,
  "y": 0.31,
  "angle": 0.73,
  "length": 0.08,
  "width": 0.015,
  "color": [0.5, 0.3, 0.2],
  "opacity": 0.75,
  "brush": "flat_oil"
}
```

V1 stroke programs are JSON files with `version`, `canvas`, `strokes`, and optional `metadata`. The schema is validated by `Source/Renderer/stroke_schema.py`.

See `Docs/structure_and_data_flow.md` for the current module boundaries and renderer data flow.

The current model direction is documented in `Docs/visual_delta_stroke_compiler.md`. In short: earlier next-stroke prediction experiments are retired as the main product path, and the next model should compile a target visual delta into editable finishing strokes.

## Repository Layout

```text
BrushWright/
  Source/
    Renderer/                  stroke schema, renderer wrappers, replay/export logic
    Synthetic/                 template generation, draft/finish splits, sample assembly
    PaintTransformerReference/ Paint Transformer model and renderer adapter
    Metrics/                   image and future stroke-level metrics

  Assets/                      local source media, including Assets/ImageCorpus/
  Outputs/                     ignored generated samples and run outputs
  Config/                      explicit project configuration
  Fixtures/                    tiny committed examples for tests and smoke runs
  Scripts/                     local run entrypoints
  Tests/                       focused automated tests
  ThirdParty/                  external notices, licenses, and local checkpoint locations
  Docs/                        architecture notes, decisions, and experiment notes
```

Folder names use `PascalCase`. File names use `snake_case`.

## Renderer Engine

BrushWright renders stroke programs with the Paint Transformer brush-mask renderer imported under `Source/PaintTransformerReference/` and exposed through `Source/Renderer/`.

To render a stroke program directly:

```bash
Scripts/run_renderer.sh Fixtures/sample_stroke_program.json Outputs/RenderSmoke
```

The renderer writes:

```text
Outputs/RenderSmoke/
  final.png
  replay.gif
  frames/frame_000001.png
  render_manifest.json
```

In VS Code:

- `buildSample` builds the default template-generated sample.
- `buildTemplateSample` prompts for seed, template, and style.
- `renderFixture` renders `Fixtures/sample_stroke_program.json`.
- `runPaintTransformer` runs `run.py` with its defaults.
- `importArtInstituteChicago` imports an Art Institute of Chicago CC0 artwork corpus.
- `runArtInstituteChicago` runs `run.py` against imported AIC images.
- `importOpenclipart` imports a small CC0/public-domain Openclipart image corpus.
- `runOpenclipart` runs `run.py` against imported Openclipart images.
- `importArtBench` imports a small ArtBench image corpus for research-only experiments.
- `runArtBench` runs `run.py` against imported ArtBench images.
- `testBrushWright` runs the unit tests.

## Synthetic Sample Builder

Build a complete pre-ML supervised sample:

```bash
python3 -m Source.Synthetic.build_sample --seed 1
```

This writes `full_program.json`, `base_strokes.json`, `finishing_strokes.json`, `draft.png`, `finished.png`, render manifests, and `sample.json` under `Outputs/Samples/sample_000001/`.

The default generator uses structured icon templates, not random stroke grids. To choose one explicitly:

```bash
python3 -m Source.Synthetic.build_sample --seed 1 --template house_icon --style flat_vector
```

Lower-level utilities are available when needed:

```bash
python3 -m Source.Synthetic.generate_programs --seed 1 --output Outputs/Samples/sample_000001/full_program.json
python3 -m Source.Synthetic.split_strokes Outputs/Samples/sample_000001/full_program.json --output-dir Outputs/Samples/sample_000001
```

## Paint Transformer Reference Samples

For more realistic synthetic data, BrushWright includes an optional Apache-2.0 Paint Transformer reference adapter under `Source/PaintTransformerReference/`. It uses Paint Transformer as a teacher/data source:

```text
source image
        |
        v
Paint Transformer checkpoint
        |
        v
ordered BrushWright stroke program
        |
        v
base_strokes + finishing_strokes supervised sample
```

The adapter imports only the inference-time model/helpers and keeps checkpoints external:

```bash
python3 run.py
```

By default, `run.py` reads:

```text
checkpoint: ThirdParty/PaintTransformer/model.pth
images:     Assets/ImageCorpus/default_input.png
output:     Outputs/PaintTransformerSamples/
split:      768 base strokes + 256 finishing strokes
draft:      native Paint Transformer frame at roughly three-fifths completion
target:     fine final native Paint Transformer output
canvas:     512x512
workers:    3
device:     cuda
```

If PyTorch is not installed for the Python you use to launch `run.py`, the runner creates a repo-local `.venv`, installs `torch`, `pillow`, and `numpy` there, then restarts itself with that environment.

The default run requests CUDA inference. If CUDA is not visible to PyTorch, fix the NVIDIA driver/device exposure or run a CPU fallback explicitly:

```bash
python3 run.py --device cpu
```

For scripted runs, the guided runner uses 3 worker processes and exports 4x the V1 stroke count by default:

```bash
python3 run.py \
  --model-path /path/to/paint_transformer_model.pth \
  --image-dir Assets/ImageCorpus \
  --workers 3 \
  --base-count 768 \
  --finishing-count 256 \
  --yes
```

If an image produces fewer accepted Paint Transformer strokes than requested, `run.py` keeps the finishing count when possible and reduces the base count for that sample instead of failing the whole run. Use `--skip-existing` to resume a failed corpus run without regenerating completed sample folders.

The lower-level module entrypoint is also available:

```bash
python3 -m Source.PaintTransformerReference.synthesize_samples \
  --model-path /path/to/paint_transformer_model.pth \
  --image-dir Assets/ImageCorpus
```

This does not replace BrushWright's task. Paint Transformer paints from source images; BrushWright learns to predict withheld finishing strokes from a draft and optional base stroke history.

Paint Transformer-backed samples render `draft.png` from roughly three-fifths native completion and a fine final `finished.png` with the Paint Transformer brush-mask renderer. Template fixtures use the same renderer path.

## Art Institute of Chicago Corpus

The Art Institute of Chicago is the preferred artwork corpus because it provides public-domain collection images and metadata under CC0, with a clean API and IIIF image service. It is a strong fit for paintings, prints, drawings, and decorative art.

Import the default 5,000-image corpus into `Assets/ImageCorpus/ArtInstituteChicago/`:

```bash
python3 -m Source.Synthetic.import_art_institute_chicago --limit 5000 --image-size 512
```

Remove near-uniform artwork borders into a separate normalized corpus:

```bash
python3 -m Source.Synthetic.preprocess_image_corpus --clear-existing
```

Then generate BrushWright samples from the cropped corpus:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/ArtInstituteChicagoCropped \
  --output-root Outputs/PaintTransformerSamples/ArtInstituteChicagoCropped \
  --limit 8 \
  --yes
```

The original imported images remain under `Assets/ImageCorpus/ArtInstituteChicago/`; preprocessing writes cropped images under `Assets/ImageCorpus/ArtInstituteChicagoCropped/` and a border-removal manifest under `Outputs/ArtInstituteChicago/`.

## Openclipart Image Corpus

Openclipart is the preferred broad external image corpus for BrushWright because uploaded artwork is released under CC0/public-domain terms. Imported files are local artifacts and are not committed.

Import a small Openclipart subset into `Assets/ImageCorpus/Openclipart/`:

```bash
python3 -m Source.Synthetic.import_openclipart --limit 64 --image-size 512
```

Then generate BrushWright samples from that corpus:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/Openclipart \
  --output-root Outputs/PaintTransformerSamples/Openclipart \
  --limit 8 \
  --yes
```

The importer uses Openclipart's JSON search API and downloads PNG renditions so `run.py` can consume them directly.


## Google Quick, Draw! Corpus

Google Quick, Draw! is an optional large doodle corpus for simple black-line artwork. The importer streams the simplified NDJSON files and renders 512x512 PNGs. The dataset is CC-BY-4.0, so keep attribution metadata with generated corpora.

Import 200 drawings per class into `Assets/ImageCorpus/GoogleQuickDraw/`:

```bash
python3 -m Source.Synthetic.import_google_quickdraw --per-class 200 --clear-existing
```

Then generate BrushWright samples from that corpus:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/GoogleQuickDraw \
  --output-root Outputs/PaintTransformerSamples/GoogleQuickDraw \
  --skip-existing \
  --yes
```

## ArtBench Image Corpus

ArtBench can be used as an optional external source-image corpus for Paint Transformer-backed sample generation, but its images are fair-use research data, not MIT-compatible assets. Imported images are local artifacts and are not committed.

Import a small shuffled ArtBench subset into `Assets/ImageCorpus/ArtBench/`:

```bash
python3 -m Source.Synthetic.import_artbench --limit 64 --image-size 512
```

Then generate BrushWright samples from that corpus:

```bash
python3 run.py \
  --image-dir Assets/ImageCorpus/ArtBench \
  --output-root Outputs/PaintTransformerSamples/ArtBench \
  --limit 8 \
  --yes
```

The importer defaults to the Hugging Face `zguo0525/ArtBench` mirror for access, while recording the official ArtBench source reference in `Outputs/ArtBench/artbench_import_manifest.json`. ArtBench is distributed for fair-use research, so keep it opt-in and external rather than treating it like project-owned or permissively licensed data.

## Non-Goals For V1

- No generic diffusion wrapper. Diffusion may later generate a masked target patch, but BrushWright's output remains editable strokes.
- No arbitrary real painting completion at first.
- No full physical paint simulation.
- No subjective unfinishedness detection.
- No large app before the data, renderer, model, and evaluator work.

## Development Principles

- Keep outputs procedural and inspectable.
- Prefer simple measurable baselines before complex models.
- Use Paint Transformer's renderer as the primary V1 painter while keeping the renderer boundary replaceable.
- Preserve existing strokes rather than overwriting the artist's intent.
- Make every generated sample reproducible from saved stroke data and seeds.

## License And References

BrushWright project code is released under the MIT License. See `LICENSE`.

The optional Paint Transformer reference subset remains under its upstream Apache-2.0 license and is documented in `ThirdParty/PaintTransformer/` and `NOTICE.md`. Pretrained checkpoints, generated datasets, downloaded source-image corpora, and local outputs are external artifacts and are not committed.

See `Docs/references.md` for upstream papers, datasets, licenses, and attribution notes used by the current research pipeline.
