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

# these constants are used to access the style_flags and face_flags
FT_FACE_FLAG_SCALABLE          = 1 << 0
FT_FACE_FLAG_FIXED_SIZES       = 1 << 1
FT_FACE_FLAG_FIXED_WIDTH       = 1 << 2
FT_FACE_FLAG_SFNT              = 1 << 3
FT_FACE_FLAG_HORIZONTAL        = 1 << 4
FT_FACE_FLAG_VERTICAL          = 1 << 5
FT_FACE_FLAG_KERNING           = 1 << 6
FT_FACE_FLAG_FAST_GLYPHS       = 1 << 7
FT_FACE_FLAG_MULTIPLE_MASTERS  = 1 << 8
FT_FACE_FLAG_GLYPH_NAMES       = 1 << 9
FT_FACE_FLAG_EXTERNAL_STREAM   = 1 << 10
FT_STYLE_FLAG_ITALIC           = 1 << 0
FT_STYLE_FLAG_BOLD             = 1 << 1

print('Num faces   :', font.num_faces)       # number of faces in file
print('Num glyphs  :', font.num_glyphs)      # number of glyphs in the face
print('Family name :', font.family_name)     # face family name
print('Syle name   :', font.style_name)      # face syle name
print('PS name     :', font.postscript_name) # the postscript name
print('Num fixed   :', font.num_fixed_sizes) # number of embedded bitmap in face

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

print('Italics       :', font.style_flags & FT_STYLE_FLAG_ITALIC          != 0)
print('Bold          :', font.style_flags & FT_STYLE_FLAG_BOLD            != 0)
print('Scalable      :', font.style_flags & FT_FACE_FLAG_SCALABLE         != 0)
print('Fixed sizes   :', font.style_flags & FT_FACE_FLAG_FIXED_SIZES      != 0)
print('Fixed width   :', font.style_flags & FT_FACE_FLAG_FIXED_WIDTH      != 0)
print('SFNT          :', font.style_flags & FT_FACE_FLAG_SFNT             != 0)
print('Horizontal    :', font.style_flags & FT_FACE_FLAG_HORIZONTAL       != 0)
print('Vertical      :', font.style_flags & FT_FACE_FLAG_VERTICAL         != 0)
print('Kerning       :', font.style_flags & FT_FACE_FLAG_KERNING          != 0)
print('Fast glyphs   :', font.style_flags & FT_FACE_FLAG_FAST_GLYPHS      != 0)
print('Mult. masters :', font.style_flags & FT_FACE_FLAG_MULTIPLE_MASTERS != 0)
print('Glyph names   :', font.style_flags & FT_FACE_FLAG_GLYPH_NAMES      != 0)

print(dir(font))

cmap = font.get_charmap()
print(font.get_kerning)
