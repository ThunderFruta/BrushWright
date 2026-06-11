# Agent Instructions

These instructions apply to all automated coding agents working in this repository.

## Project Direction

BrushWright is a stroke-level painting completion project. The central task is:

```text
input: rough draft image and optional base stroke history
output: missing finishing strokes
```

Do not reduce the project to a generic image-to-image or inpainting wrapper. The repo should stay centered on stroke programs, renderer adapters, synthetic data, supervised finishing-stroke prediction, and evaluation.

## Naming Conventions

- Use `PascalCase` for folders.
- Use `snake_case` for source files, scripts, config files, and generated data files.
- Use `SCREAMING_SNAKE_CASE` for constants, environment variables, and appropriate global configuration values.
- Use `PascalCase` for classes and typed data models.
- Use `snake_case` for functions, methods, variables, and JSON fields.

Examples:

```text
Source/Renderer/stroke_schema.py
Source/Synthetic/generate_programs.py
Model/stroke_decoder.py
Source/Metrics/render_loss.py
```

```python
DEFAULT_FINISHING_STROKES = 64

class StrokeProgram:
    pass

def render_strokes(stroke_program):
    pass
```

## Architecture Boundaries

Keep these responsibilities separate:

- `Source/Synthetic/`: synthetic stroke program generation, draft/finish splits, datasets, augmentation.
- `Source/Renderer/`: stroke schema, renderer adapters, compositing, replay, export.
- `Source/Metrics/`: stroke-level, image-level, style, preservation, and report metrics.
- `Source/PaintTransformerReference/`: imported Paint Transformer model and renderer helpers used for synthetic data.
- `Model/`: encoders, decoders, training, inference.
- `App/`: local inspection tools, comparison views, replay UI.
- `Docs/`: design notes, experiments, model cards, and technical decisions.
- `Assets/`: local source media and manual input images.
- `Outputs/`: generated datasets and local run artifacts only.

Avoid cross-module imports that make synthetic generation depend on app code or model training depend on UI code.

## V1 Constraints

Default to the V1 problem unless the user explicitly changes scope:

- Canvas size: `512x512`
- Base strokes: `192`
- Finishing strokes: `64`
- Target: predict withheld finishing strokes
- Training: supervised stroke prediction
- Renderer: Paint Transformer brush-mask renderer through `Source/Renderer/`

## Engineering Standards

- Prefer deterministic generation with explicit seeds.
- Save enough metadata to reproduce every synthetic sample.
- Store generated samples as data artifacts, not hand-authored source.
- Keep configs explicit and versioned.
- Add focused tests for schema validation, deterministic generation, renderer adapter behavior, and metrics.
- Avoid introducing large dependencies without documenting why.

## Editing Rules

- Do not rename architectural folders away from `PascalCase`.
- Do not introduce camelCase file names.
- Do not use magic constants inline when they represent project settings.
- Do not commit generated datasets unless they are tiny fixtures.
- Do not put implementation code under `Assets/` or `Outputs/`.
- Do not recreate the removed legacy data folder or compatibility imports.
- Do not overwrite user-created experiments or output artifacts unless explicitly asked.

## Documentation Expectations

When adding a major module, include:

- what the module owns
- expected inputs and outputs
- reproducibility assumptions
- known limitations
- a minimal command example once runnable

Keep documentation practical and current.
