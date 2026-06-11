# Metrics

`Source/Metrics/` owns simple checks and comparisons for rendered images and future stroke-level evaluation.

The current implementation provides basic RGB image metrics for validating renderer and sample outputs.

## Example

```bash
python3 -m Source.Metrics.image_metrics Outputs/Samples/sample_000001/draft.png Outputs/Samples/sample_000001/finished.png
```
