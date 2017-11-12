"""
===============
Face properties
===============

This is a demo script to show you how to use all the properties of a Face
object.  These describe global font properties.  For individual character
metrics, use the Glyph object, as loaded from the glyph attribute after calling
load_char.
"""
import matplotlib
from matplotlib import font_manager, _ft2


fname = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans-Oblique.ttf"
font = font_manager.get_font(fname)

print("Faces in file          :", font.num_faces)
print("Glyphs in face         :", font.num_glyphs)
print("Family name            :", font.family_name)
print("Style name             :", font.style_name)
print("Postscript name        :", font.get_postscript_name())
print("Embedded bitmap strikes:", font.num_fixed_sizes)

if font.face_flags & _ft2.FACE_FLAG_SCALABLE:
    print('Global bbox (xmin, ymin, xmax, ymax):', font.bbox)
    print('Font units per EM                   :', font.units_per_EM)
    print('Ascender (pixels)                   :', font.ascender)
    print('Descender (pixels)                  :', font.descender)
    print('Height (pixels)                     :', font.height)
    print('Max horizontal advance              :', font.max_advance_width)
    print('Max vertical advance                :', font.max_advance_height)
    print('Underline position                  :', font.underline_position)
    print('Underline thickness                 :', font.underline_thickness)

for style in ['Style flag italic',
              'Style flag bold']:
    flag = getattr(_ft2, style.replace(' ', '_').upper()) - 1
    print('%-26s:' % style, bool(font.style_flags & flag))

for style in ['Face flag scalable',
              'Face flag fixed sizes',
              'Face flag fixed width',
              'Face flag SFNT',
              'Face flag horizontal',
              'Face flag vertical',
              'Face flag kerning',
              'Face flag fast glyphs',
              'Face flag multiple masters',
              'Face flag glyph names',
              'Face flag external stream']:
    flag = getattr(_ft2, style.replace(' ', '_').upper())
    print('%-26s:' % style, bool(font.face_flags & flag))
