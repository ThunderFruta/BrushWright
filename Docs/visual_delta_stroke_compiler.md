# Visual Delta Stroke Compiler Direction

## Summary

BrushWright should move away from predicting the next withheld Paint Transformer stroke chunk as the main product path.
The current V1/V2 predictors proved the plumbing, but they did not produce visible improvement: the stroke loss could drop while rendered predictions barely changed the draft.

The next direction is a visual-delta-to-strokes pipeline:

```text
draft image + target image/patch + edit mask + optional intent
        |
        v
visual delta / error map
        |
        v
stroke compiler
        |
        v
editable finishing strokes
        |
        v
renderer + visual validation
```

The target image can come from synthetic `finished.png` at first. Later it can come from a diffusion inpainted patch, a user paintover, a PSD layer edit, a reference image, or a vision-language planning step.

This keeps BrushWright centered on stroke output. Diffusion and language models are optional target generators, not the final product.

## Retired Pipelines

The following pipelines are now retired as product directions:

- **Template-only synthetic completion**: useful for renderer/schema tests, but too simple to define the product.
- **PaintTransformer next-stroke prediction**: useful for data bootstrapping, but target stroke order is a teacher artifact and does not guarantee visual improvement.
- **Stroke-only chunk prediction from `base_strokes.json`**: underdetermined without image context.
- **Draft/goal/error-conditioned chunk prediction with generic learned slots**: improved the interface, but still failed the overfit visual gate because output slots were not reliably bound to missing regions.

Retired does not mean deleted. Keep the code and checkpoints as experiments and regression fixtures until replacement paths cover the same plumbing.

## New Pipeline Contract

The model should learn this task:

```text
input:
  draft patch/image
  target patch/image
  edit mask
  error map = abs(target - draft)
  optional base stroke history
  optional text intent

output:
  added_strokes.json
```

The output strokes must render over the draft and move it closer to the target. The training objective and evaluation must use rendered visual improvement, not only per-stroke numeric loss.

For synthetic training, the first target source remains existing data:

```text
draft.png        = Paint Transformer native draft frame / render(base_strokes)
target.png       = resized original source image
finished.png     = Paint Transformer final render kept as reference metadata
edit mask        = threshold(abs(target.png - draft.png)) with cleanup
target strokes   = finishing_strokes.json when useful for supervised hints
```

For future interactive use, the target source can be:

```text
prompt/reference/PSD/user request
        |
        v
vision-language planner
        |
        v
masked diffusion or user paintover target patch
        |
        v
visual delta stroke compiler
```

## Model Direction

The next model should be a stroke compiler, not a next-stroke predictor.

Required behavior:

- Anchor predicted strokes to error-mask regions or target patch features.
- Predict strokes for local visual deltas, not global sequence continuation.
- Support patch-level inference so the system can edit one region without repainting the canvas.
- Preserve the existing renderer boundary and stroke schema.
- Treat `finishing_strokes.json` as optional teacher signal, not the only measure of success.

Recommended first implementation:

- Build a V3 dataset item around `(draft_image, target_image, error_map, edit_mask, optional finishing_strokes)`.
- Train on `64x64` patches sampled from changed regions plus some unchanged negatives.
- Use supervised stroke losses only where target strokes are available, but gate progress by rendered image metrics.
- Add a one-sample overfit gate that must visibly beat the draft before full training is trusted.

## Evaluation Gates

A checkpoint is not useful unless it passes visual gates:

- `predicted_vs_target` rendered image distance is lower than `draft_vs_target`.
- Predicted strokes change more than a minimum pixel threshold inside the edit mask.
- Pixel changes outside the edit mask stay below a preservation threshold.
- One-sample overfit produces a visible improvement before any full training run is accepted.
- Test exports include `draft.png`, `target.png`, `predicted.png`, `comparison.png`, `added_strokes.json`, and diagnostics.

Scalar stroke loss is still useful for debugging, but it must not choose the best checkpoint by itself.

## Implementation Notes

- Keep old V1/V2 code runnable for comparison, but label it as experimental/retired in commands and docs.
- Put new training/inference code under `Source/Model/` and keep renderer calls behind `Source/Renderer/`.
- Do not make diffusion a hard dependency for V3. The first target provider is existing `finished.png`.
- Add diffusion later as a `TargetProvider`-style adapter that writes a target patch and mask.
- Add natural-language planning after the stroke compiler can already convert a target visual delta into useful strokes.

Minimal future command shape:

```bash
python3 -m Source.Model.train_visual_delta_strokes --data-root Data --device cuda
```

Minimal learned export shape:

```bash
python3 -m Source.Model.export_visual_delta_predictions --split Test --device cuda
```

The current default learned compiler is `VisualDeltaStrokeCompilerV8UsableV1Large`.
It is the usable-V1 scale target for a roughly 48GB GPU class, not the earlier
prototype-scale V7 setting. The default training run uses 128px overlapping
patches on the 512x512 V1 canvas, a 768-wide transformer decoder, 10 decoder
layers, 12 attention heads, 3072 feed-forward width, a 16x16 visual feature
grid, and 512 stroke proposals per patch. The 512 proposal slots map to an
11x11 coarse anchor grid plus a 16x32 detail grid, truncated to the active
proposal count. Start with the default batch size of 4; raise it only after
checking real VRAM headroom on the cloud GPU.

Its decoder ranges are large enough to represent the observed output/detail-pair
finishing stroke distribution, so export cannot collapse to all sub-threshold
dot strokes. Checkpoint selection remains visual-only: `best.pt` is written
only after rendered validation passes the improvement gates.
Export refuses non-`best` checkpoints by default; use
`--allow-visual-failed-checkpoint` only for debugging a failed `latest.pt`.

## Remote GPU Workflow

Use `Scripts/remote_gpu.sh` to keep a cloud GPU checkout synchronized over SSH
with incremental `rsync` transfers. The remote path persists between runs, so
normal iterations only push changed source files and pull new checkpoints or
outputs.

The current RunPod endpoint is wrapped by `Scripts/runpod_gpu.sh`, which defaults
to:

```text
remote: eq39y5ydoccbct-64411cd6@ssh.runpod.io
key:    ~/.ssh/id_ed25519
path:   /workspace/BrushWright
```

RunPod setup and training:

```bash
Scripts/runpod_gpu.sh check
Scripts/runpod_gpu.sh push-all
Scripts/runpod_gpu.sh train
Scripts/runpod_gpu.sh pull-artifacts
```

Open a shell on the pod:

```bash
Scripts/runpod_gpu.sh shell
```

The `ssh.runpod.io` gateway for this pod requires a TTY. The RunPod wrapper
sets `BRUSHWRIGHT_SSH_FORCE_TTY=1`, so command execution and code sync use a
PTY-compatible path. If the pod exposes a direct SSH port later, prefer the
generic `remote_gpu.sh` path with `BRUSHWRIGHT_REMOTE` and `BRUSHWRIGHT_SSH_OPTS`
because it can use normal `rsync` for large data.

Initial setup, including generated data and the Paint Transformer checkpoint:

```bash
BRUSHWRIGHT_REMOTE=root@GPU_HOST \
BRUSHWRIGHT_REMOTE_DIR=/workspace/BrushWright \
Scripts/remote_gpu.sh push-all
```

Train on the remote GPU with the V8 large defaults:

```bash
BRUSHWRIGHT_REMOTE=root@GPU_HOST \
BRUSHWRIGHT_REMOTE_DIR=/workspace/BrushWright \
Scripts/remote_gpu.sh train --device cuda --visual-validation-device cuda
```

For later code-only iterations, omit data sync:

```bash
BRUSHWRIGHT_REMOTE=root@GPU_HOST Scripts/remote_gpu.sh train
```

Pull checkpoints and rendered outputs back:

```bash
BRUSHWRIGHT_REMOTE=root@GPU_HOST Scripts/remote_gpu.sh pull-artifacts
```

Set `BRUSHWRIGHT_SYNC_DATA=1` on `train` only when regenerated `Data/`,
`Assets/ImageCorpus/`, or `ThirdParty/PaintTransformer/` files need to be pushed
again.

## Classical Greedy Optimizer

The current non-neural baseline is `GreedyStrokeOptimizerV1`. It is a comparison
fixture, not the primary predictor path. It owns classical
stroke selection for cases where the learned compiler is not yet visually
trustworthy:

```text
draft.png + target.png
        |
        v
high-error anchors + multi-size stroke candidates
        |
        v
largest candidate tier that improves rendered target distance
        |
        v
added_strokes.json
```

It uses a fast in-memory stroke approximation to search candidates, then writes
editable BrushWright strokes and renders the final `predicted.png` through the
Paint Transformer renderer adapter. The default target mode is the resized
source image, and `finished-image` remains supported for synthetic validation.
After the first coarse strokes, the optimizer regularly tries smaller detail
tiers with a lower improvement threshold so details do not wait for every broad
fill candidate to be exhausted.

Minimal command:

```bash
python3 -m Source.Model.export_greedy_stroke_optimizer --split Test --sample-id paint_transformer_000005 --target-mode source-image
```

## Assumptions

- BrushWright remains an editable-stroke system, not a generic image-to-image wrapper.
- Diffusion is useful as a target patch generator, but not required for the first V3 training loop.
- Vision-language planning is valuable for real user intent, but it should not be added before the visual delta stroke compiler works.
- Existing Paint Transformer samples remain useful as synthetic data and renderer fixtures, even though next-stroke prediction is retired as the main direction.
