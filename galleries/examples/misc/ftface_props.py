"""
===============
Font properties
===============

This example lists the attributes of an `.FT2Font` object, which describe
global font properties.  For individual character metrics, use the `.Glyph`
object, as returned by `.load_char`.
"""

import os

import matplotlib
import matplotlib.ft2font as ft

font = ft.FT2Font(
    # Use a font shipped with Matplotlib.
    os.path.join(matplotlib.get_data_path(),
                 'fonts/ttf/DejaVuSans-Oblique.ttf'))

print('Num faces:  ', font.num_faces)        # number of faces in file
print('Num glyphs: ', font.num_glyphs)       # number of glyphs in the face
print('Family name:', font.family_name)      # face family name
print('Style name: ', font.style_name)       # face style name
print('PS name:    ', font.postscript_name)  # the postscript name
print('Num fixed:  ', font.num_fixed_sizes)  # number of embedded bitmaps

# the following are only available if face.scalable
if font.scalable:
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
