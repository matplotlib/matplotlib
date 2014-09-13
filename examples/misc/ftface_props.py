#!/usr/bin/env python

from __future__ import print_function
"""
This is a demo script to show you how to use all the properties of an
FT2Font object.  These describe global font properties.  For
individual character metrices, use the Glyp object, as returned by
load_char
"""
import matplotlib
from matplotlib.ft2font import FT2Font

#fname = '/usr/local/share/matplotlib/VeraIt.ttf'
fname = matplotlib.get_data_path() + '/fonts/ttf/VeraIt.ttf'
#fname = '/usr/local/share/matplotlib/cmr10.ttf'

font = FT2Font(fname)

# these globals are used to access the style_flags and face_flags
FT_STYLE_FLAGS = (
    ('Italics', 0),
    ('Bold', 1)
)

FT_FACE_FLAGS = (
    ('Scalable', 0),
    ('Fixed sizes', 1),
    ('Fixed width', 2),
    ('SFNT', 3),
    ('Horizontal', 4),
    ('Vertical', 5),
    ('Kerning', 6),
    ('Fast glyphs', 7),
    ('Mult. masters', 8),
    ('Glyph names', 9),
    ('External stream', 10)
)


print('Num faces   :', font.num_faces)       # number of faces in file
print('Num glyphs  :', font.num_glyphs)      # number of glyphs in the face
print('Family name :', font.family_name)     # face family name
print('Syle name   :', font.style_name)      # face syle name
print('PS name     :', font.postscript_name)  # the postscript name
# number of embedded bitmap in face
print('Num fixed   :', font.num_fixed_sizes)

# the following are only available if face.scalable
if font.scalable:
    # the face global bounding box (xmin, ymin, xmax, ymax)
    print('Bbox                :', font.bbox)
    # number of font units covered by the EM
    print('EM                  :', font.units_per_EM)
    # the ascender in 26.6 units
    print('Ascender            :', font.ascender)
    # the descender in 26.6 units
    print('Descender           :', font.descender)
    # the height in 26.6 units
    print('Height              :', font.height)
    # maximum horizontal cursor advance
    print('Max adv width       :', font.max_advance_width)
    # same for vertical layout
    print('Max adv height      :', font.max_advance_height)
    # vertical position of the underline bar
    print('Underline pos       :', font.underline_position)
    # vertical thickness of the underline
    print('Underline thickness :', font.underline_thickness)

for desc, val in FT_STYLE_FLAGS:
    print('%-16s:' % desc, bool(font.style_flags & (1 << val)))
for desc, val in FT_FACE_FLAGS:
    print('%-16s:' % desc, bool(font.style_flags & (1 << val)))

print(dir(font))

cmap = font.get_charmap()
print(font.get_kerning)
