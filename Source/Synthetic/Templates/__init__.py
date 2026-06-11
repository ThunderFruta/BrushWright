"""Structured synthetic drawing templates."""

from Source.Synthetic.Templates.geometric_badge import generate_geometric_badge
from Source.Synthetic.Templates.house_icon import generate_house_icon
from Source.Synthetic.Templates.face_icon import generate_face_icon
from Source.Synthetic.Templates.flower_icon import generate_flower_icon
from Source.Synthetic.Templates.tree_icon import generate_tree_icon


TEMPLATES = {
    "house_icon": generate_house_icon,
    "tree_icon": generate_tree_icon,
    "flower_icon": generate_flower_icon,
    "face_icon": generate_face_icon,
    "geometric_badge": generate_geometric_badge,
}

TEMPLATE_NAMES = tuple(TEMPLATES.keys())

