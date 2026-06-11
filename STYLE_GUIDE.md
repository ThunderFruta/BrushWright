# Style Guide

This guide defines repository structure, naming, data conventions, and code style for BrushWright.

## Naming

Use these conventions consistently:

| Item | Convention | Example |
|---|---|---|
| Folders | `PascalCase` | `Source/`, `Renderer/`, `Synthetic/` |
| Files | `snake_case` | `stroke_schema.py`, `render_loss.py` |
| Python classes | `PascalCase` | `StrokeProgram` |
| Functions | `snake_case` | `render_strokes` |
| Variables | `snake_case` | `finish_strokes` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_FINISHING_STROKES` |
| JSON fields | `snake_case` | `base_strokes` |
| CLI flags | `kebab-case` | `--canvas-size` |

Use constants for shared configuration:

```python
DEFAULT_CANVAS_SIZE = 512
DEFAULT_BASE_STROKES = 192
DEFAULT_FINISHING_STROKES = 64
DEFAULT_BRUSH_FAMILY = "flat_oil"
```

## Folder Layout

Preferred top-level layout:

```text
BrushWright/
  App/
  Assets/
  Config/
  Docs/
  Fixtures/
  Model/
  Outputs/
  Scripts/
  Source/
    Metrics/
    PaintTransformerReference/
    Renderer/
    Synthetic/
  Tests/
  ThirdParty/
```

Keep implementation code under `Source/`. Keep source media under `Assets/`. Keep generated datasets, renderer smoke outputs, and run artifacts under `Outputs/`.

## File Naming

Good:

```text
Source/Renderer/stroke_schema.py
Source/Renderer/render_program.py
Source/Synthetic/generate_programs.py
Source/Synthetic/split_strokes.py
Source/Metrics/palette_match.py
Model/stroke_decoder.py
```

Avoid:

```text
renderer/strokeSchema.py
data/GeneratePrograms.py
model/stroke-decoder.py
metrics/PaletteMatch.py
```

## Stroke Schema

Use normalized coordinates and dimensions unless a module explicitly requires pixel coordinates.

Recommended V1 stroke:

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

Guidelines:

- `x`, `y`, `length`, and `width` are normalized to canvas size.
- `angle` is normalized to turns or explicitly documented radians; do not mix both.
- `color` is RGB float `[0.0, 1.0]` unless otherwise documented.
- `opacity` is float `[0.0, 1.0]`.
- `brush` is a stable string identifier.

## Generated Artifacts

Generated samples should be reproducible from metadata and should live under `Outputs/` by default.

Recommended sample shape:

```json
{
  "sample_id": "000001",
  "seed": 12345,
  "canvas": {
    "width": 512,
    "height": 512
  },
  "style": "flat_oil",
  "base_strokes": [],
  "finishing_strokes": [],
  "draft_image": "draft.png",
  "finished_image": "finished.png"
}
```

Large generated data should not be committed. Keep only tiny fixtures for tests and examples.

## Code Style

- Prefer small modules with clear ownership.
- Keep synthetic generation, renderer, model, metrics, and app code decoupled.
- Prefer typed data structures for stroke programs and samples.
- Keep random generation deterministic through explicit seeds.
- Validate stroke data at module boundaries.
- Make scripts runnable with clear CLI arguments.

## Constants

Use `SCREAMING_SNAKE_CASE` for:

- canvas defaults
- stroke count limits
- supported brush names
- filesystem defaults
- model dimension defaults
- metric thresholds

Example:

```python
SUPPORTED_BRUSHES = ("flat_oil", "paint_transformer")
DEFAULT_CANVAS_SIZE = 512
DEFAULT_BASE_STROKES = 192
DEFAULT_FINISHING_STROKES = 64
```

Do not use `SCREAMING_SNAKE_CASE` for ordinary local variables.

## Documentation Tone

Documentation should be direct and implementation-oriented. Prefer concrete data shapes, module responsibilities, and runnable examples over broad product language.

## Testing Priorities

Prioritize tests for:

- stroke schema validation
- deterministic synthetic generation
- base/finish stroke splitting
- renderer adapter reproducibility
- metric sanity checks
- small end-to-end fixture generation
