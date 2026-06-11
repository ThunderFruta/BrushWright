# Paint Transformer Notice

BrushWright imports a minimal inference-time subset of the unofficial PyTorch Paint Transformer implementation for synthetic data generation.

- Upstream: `https://github.com/Huage001/PaintTransformer`
- Paper: `Paint Transformer: Feed Forward Neural Painting with Stroke Prediction`
- License: Apache-2.0
- Local use: optional reference model for converting image corpus inputs into ordered stroke programs

Only the `Painter` model, inference canvas update helpers, and required brush masks are used. Training code, demos, sample images, and checkpoints are intentionally not vendored.

The pretrained checkpoint is not committed. Pass it explicitly to:

```bash
python3 run.py --model-path /path/to/model.pth --image Assets/ImageCorpus/example.jpg
```

Or place it at the default local path and run:

```bash
python3 run.py
```

Default checkpoint path:

```text
ThirdParty/PaintTransformer/model.pth
```
