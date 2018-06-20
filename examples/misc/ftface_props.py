"""
============
Ftface Props
============

This is a demo script to show you how to use all the properties of an
FT2Font object.  These describe global font properties.  For
individual character metrics, use the Glyph object, as returned by
load_char
"""
import matplotlib
import matplotlib.ft2font as ft


#fname = '/usr/local/share/matplotlib/VeraIt.ttf'
fname = matplotlib.get_data_path() + '/fonts/ttf/DejaVuSans-Oblique.ttf'
#fname = '/usr/local/share/matplotlib/cmr10.ttf'

font = ft.FT2Font(fname)

print('Num faces   :', font.num_faces)        # number of faces in file
print('Num glyphs  :', font.num_glyphs)       # number of glyphs in the face
print('Family name :', font.family_name)      # face family name
print('Style name  :', font.style_name)       # face style name
print('PS name     :', font.postscript_name)  # the postscript name
print('Num fixed   :', font.num_fixed_sizes)  # number of embedded bitmap in face

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

for style in ('Italic',
              'Bold',
              'Scalable',
              'Fixed sizes',
              'Fixed width',
              'SFNT',
              'Horizontal',
              'Vertical',
              'Kerning',
              'Fast glyphs',
              'Multiple masters',
              'Glyph names',
              'External stream'):
    bitpos = getattr(ft, style.replace(' ', '_').upper()) - 1
    print('%-17s:' % style, bool(font.style_flags & (1 << bitpos)))

print(dir(font))

print(font.get_kerning)
