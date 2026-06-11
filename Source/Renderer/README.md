# Renderer

`Source/Renderer/` owns stroke program validation and conversion from BrushWright stroke JSON into rendered image artifacts.

## Inputs

The V1 renderer accepts a stroke program JSON object with:

- `version`: currently `1`
- `canvas`: `width` and `height`, defaulting to `512x512`
- `strokes`: ordered finishing or full-program strokes
- `metadata`: optional reproducibility details

Stroke coordinates and dimensions are normalized floats. `angle` is normalized turns, where `1.0` is one full rotation. `color` is RGB float `[0.0, 1.0]`.

## Outputs

The renderer writes:

```text
final.png
replay.gif
frames/frame_000001.png
render_manifest.json
```

Each frame is the canvas after one completed stroke. `final.png` is a copy of the last frame.

## Reproducibility

Rendering is deterministic for a fixed stroke program, Paint Transformer renderer implementation, PyTorch version, and output canvas size. The renderer does not generate strokes or mutate the input program.

## Known Limitations

The V1 renderer maps BrushWright's normalized stroke schema into Paint Transformer's rectangular brush-mask parameterization. The `brush` field is validated and preserved, but multiple brush presets are not yet implemented.

## Example

```bash
Scripts/run_renderer.sh Fixtures/sample_stroke_program.json Outputs/RenderSmoke
```
