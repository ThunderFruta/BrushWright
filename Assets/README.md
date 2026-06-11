# Assets

`Assets/` stores local reference media and real-world test inputs used to inspect BrushWright behavior outside the synthetic data pipeline.

Use this folder for:

- rough draft images from real drawing or painting workflows
- reference finished images for manual comparison
- small local evaluation sets
- renderer or app inspection inputs that are not generated fixtures

Do not use this folder for generated training datasets. Synthetic samples belong under a generated data output path, not committed source.

## Layout

```text
Assets/
  README.md
  drafts/
  references/
  notes/
```

Keep committed assets small and intentional. Large, private, or experimental real-world files should remain local unless explicitly approved for commit.
