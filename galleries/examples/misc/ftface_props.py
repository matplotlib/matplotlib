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

# Use a font shipped with Matplotlib.
font_path = os.path.join(
    matplotlib.get_data_path(),
    'fonts/ttf/DejaVuSans-Oblique.ttf'
)

font = ft.FT2Font(font_path)

print('Num instances:  ', font.num_named_instances)  # number of named instances in file
print('Num faces:      ', font.num_faces)            # number of faces in file
print('Num glyphs:     ', font.num_glyphs)           # number of glyphs in the face
print('Family name:    ', font.family_name)          # face family name
print('Style name:     ', font.style_name)           # face style name
print('PS name:        ', font.postscript_name)      # the postscript name
print('Num fixed:      ', font.num_fixed_sizes)      # number of embedded bitmaps

# the face global bounding box (xmin, ymin, xmax, ymax)
print('Bbox:               ', font.bbox)
# number of font units covered by the EM
print('EM:                 ', font.units_per_EM)
# the ascender in 26.6 units
print('Ascender:           ', font.ascender)
# the descender in 26.6 units
print('Descender:          ', font.descender)
# the height in 26.6 units
print('Height:             ', font.height)
# maximum horizontal cursor advance
print('Max adv width:      ', font.max_advance_width)
# same for vertical layout
print('Max adv height:     ', font.max_advance_height)
# vertical position of the underline bar
print('Underline pos:      ', font.underline_position)
# vertical thickness of the underline
print('Underline thickness:', font.underline_thickness)

for flag in ft.StyleFlags:
    name = flag.name.replace('_', ' ').title() + ':'
    print(f"{name:17}", flag in font.style_flags)

for flag in ft.FaceFlags:
    name = flag.name.replace('_', ' ').title() + ':'
    print(f"{name:17}", flag in font.face_flags)

# ── Visualise font metrics ────────────────────────────────────────────────────
# Normalise all metrics to units_per_EM so values are in the range [-1, 1].
# This figure is used by Sphinx Gallery to auto-generate the gallery thumbnail.
u = font.units_per_EM
asc = font.ascender / u
desc = font.descender / u
bbox_ymax = font.bbox[3] / u
bbox_ymin = font.bbox[1] / u
ul_pos = font.underline_position / u
ul_thick = font.underline_thickness / u

fig, ax = plt.subplots(figsize=(8, 6))

# Metric lines drawn FIRST (lower zorder) so text renders on top of them.
metrics = [
    ("bbox top (ymax)",    bbox_ymax, "tab:green"),
    ("ascender",           asc,       "tab:blue"),
    ("baseline (y=0)",     0,         "black"),
    ("underline_position", ul_pos,    "tab:orange"),
    ("descender",          desc,      "tab:red"),
    ("bbox bottom (ymin)", bbox_ymin, "tab:purple"),
]

# Lines span from left edge to 72% of axes width — crossing through the glyph.
# Labels sit at 75%, clearly to the right of the lines.
for label, y, color in metrics:
    ax.plot(
        [0.02, 0.72], [y, y],
        color=color, linewidth=1.5, linestyle='--', alpha=0.9, zorder=2)
    # default position
    y_pos = y

    # adjust only bbox labels
    if "bbox top" in label:
        y_pos = y - 0.015
    elif "bbox bottom" in label:
        y_pos = y + 0.015

    ax.text(
        0.75, y_pos, label, color=color, va='center',
        fontsize=9, fontweight='medium', ha='left', zorder=2)

# Underline thickness — shaded band between underline_position and its lower edge.
ax.fill_between([0.02, 0.72],
                ul_pos - ul_thick,
                ul_pos,
                color='tab:orange',
                alpha=0.22,
                label=f'underline_thickness = {font.underline_thickness}',
                zorder=1)

# Bounding box (font.bbox) as a rectangle. Drawn after lines, before text.
ax.add_patch(plt.Rectangle(
    (0.02, bbox_ymin), 0.70, (bbox_ymax - bbox_ymin),
    fill=False, edgecolor='black', linestyle='-',
    linewidth=1.5, alpha=0.6, zorder=3,
    label='font.bbox'
))

# Render "Ag" on top of everything — zorder=10 ensures no line covers the text.
# 'A' shows ascender/cap-height, 'g' shows descender.
fp = FontProperties(fname=font_path)
ax.text(0.30, 0.0, "Ag", fontproperties=fp, fontsize=150,
        va='baseline', ha='center', color='black', zorder=10)

ax.set_xlim(0, 1.35)
ax.set_ylim(bbox_ymin - 0.10, bbox_ymax + 0.15)
ax.set_title(
    f"Font metrics — {font.family_name} {font.style_name}\n"
    f"(values normalised to units_per_EM = {font.units_per_EM})",
    fontsize=11.5, pad=15
)
ax.legend(fontsize=8, loc='lower right', frameon=False)
ax.axis('off')
plt.tight_layout(pad=1.5)
plt.show()
