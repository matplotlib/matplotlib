"""
===============
Font properties
===============

This example lists the attributes of an `.FT2Font` object, which describe
global font properties.  For individual character metrics, use the `.Glyph`
object, as returned by `.load_char`.
"""

import os

import matplotlib.pyplot as plt

import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.ft2font as ft
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.textpath import TextPath
import matplotlib.transforms

# Use a font shipped with Matplotlib.
font_path = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Oblique.ttf")

font = ft.FT2Font(font_path)

print("Num instances:  ", font.num_named_instances)  # number of named instances in file
print("Num faces:      ", font.num_faces)  # number of faces in file
print("Num glyphs:     ", font.num_glyphs)  # number of glyphs in the face
print("Family name:    ", font.family_name)  # face family name
print("Style name:     ", font.style_name)  # face style name
print("PS name:        ", font.postscript_name)  # the postscript name
print("Num fixed:      ", font.num_fixed_sizes)  # number of embedded bitmaps
print("Bbox:               ", font.bbox)  # global bounding box (xmin, ymin, xmax, ymax)
print("EM:                 ", font.units_per_EM)  # font units per EM
print("Ascender:           ", font.ascender)  # the ascender in 26.6 units
print("Descender:          ", font.descender)  # the descender in 26.6 units
print("Height:             ", font.height)  # the height in 26.6 units
print("Max adv width:      ", font.max_advance_width)  # max horizontal advance
print("Max adv height:     ", font.max_advance_height)  # same for vertical layout
print("Underline pos:      ", font.underline_position)  # underline bar position
print("Underline thickness:", font.underline_thickness)  # underline thickness

for flag in ft.StyleFlags:
    name = flag.name.replace("_", " ").title() + ":"
    print(f"{name:17}", flag in font.style_flags)

for flag in ft.FaceFlags:
    name = flag.name.replace("_", " ").title() + ":"
    print(f"{name:17}", flag in font.face_flags)

# Normalise all vertical metrics to units_per_EM so all y-values sit in [-1, 1].
u = font.units_per_EM
asc = font.ascender / u
desc = font.descender / u
bbox_ymax = font.bbox[3] / u
bbox_ymin = font.bbox[1] / u
bbox_xmin = font.bbox[0] / u
bbox_xmax = font.bbox[2] / u
ul_pos = font.underline_position / u
ul_thick = font.underline_thickness / u

fig, ax = plt.subplots(figsize=(9.8, 6))

fp = FontProperties(fname=font_path)
tp = TextPath((0, 0), "Água", size=1, prop=fp)
text_bb = tp.get_extents()

# Align the glyph's left edge with bbox_xmin.
x_offset = bbox_xmin - text_bb.x0

# Lines, rectangle and labels are derived from the true font bbox.
LINE_X0 = bbox_xmin
LINE_X1 = bbox_xmax
LABEL_X = LINE_X1 + 0.08  # metric labels start here

metrics = [
    ("bbox top (ymax)", bbox_ymax, "tab:green"),
    ("ascender", asc, "tab:blue"),
    ("y = 0 (origin)", 0, "black"),
    ("underline_position", ul_pos, "tab:orange"),
    ("descender", desc, "tab:red"),
    ("bbox bottom (ymin)", bbox_ymin, "tab:purple"),
]

for label, y, color in metrics:
    ax.plot(
        [LINE_X0, LINE_X1],
        [y, y],
        color=color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.9,
        zorder=2,
    )
    # Nudge bbox-edge labels slightly away from the rectangle border.
    y_text = (
        y - 0.015 if "bbox top" in label else y + 0.015 if "bbox bottom" in label else y
    )
    ax.text(
        LABEL_X,
        y_text,
        label,
        color=color,
        va="center",
        ha="left",
        fontsize=9,
        fontweight="medium",
        zorder=2,
    )

# Underline thickness: shaded band from (ul_pos − ul_thick) to ul_pos.
ax.fill_between(
    [LINE_X0, LINE_X1],
    ul_pos - ul_thick,
    ul_pos,
    color="tab:orange",
    alpha=0.22,
    label=f"underline_thickness = {font.underline_thickness}",
    zorder=1,
)

# font.bbox visualised as a rectangle using the true font x/y bounds.
ax.add_patch(
    Rectangle(
        (bbox_xmin, bbox_ymin),
        bbox_xmax - bbox_xmin,
        bbox_ymax - bbox_ymin,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        linestyle="-",
        alpha=0.6,
        zorder=3,
        label="font.bbox",
    )
)

# Glyph path — translate only (scale = 1.0 implicit); high zorder so it sits
# on top of the reference lines.
ax.add_patch(
    PathPatch(
        tp,
        transform=matplotlib.transforms.Affine2D().translate(x_offset, 0)
        + ax.transData,
        color="black",
        zorder=10,
    )
)

# x-limit: start at the bbox left edge and leave room for labels.
ax.set_xlim(LINE_X0, LABEL_X + 0.75)
ax.set_ylim(bbox_ymin - 0.10, bbox_ymax + 0.15)
ax.set_title(
    f"Font metrics — {font.family_name} {font.style_name}", fontsize=11.5, pad=15
)
ax.legend(
    fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=2
)
ax.axis("off")
plt.tight_layout(pad=1.5)
plt.show()
