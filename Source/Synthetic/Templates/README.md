# Templates

`Source/Synthetic/Templates/` owns structured synthetic icon templates for V1 training samples.

Templates emit role-separated strokes:

```python
{
    "template": "house_icon",
    "style": "flat_vector",
    "base_strokes": [],
    "finishing_strokes": []
}
```

`base_strokes` define the main icon silhouette and large forms. `finishing_strokes` define meaningful withheld details such as highlights, windows, trim, eyes, petals, branches, or inner accents.

Available V1 templates:

- `house_icon`
- `tree_icon`
- `flower_icon`
- `face_icon`
- `geometric_badge`

Templates use normalized V1 stroke fields only. Styles are selected from `Config/stroke_styles.json`.

