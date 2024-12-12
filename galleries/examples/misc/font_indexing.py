"""
=============
Font indexing
=============

This example shows how the font tables relate to one another.
"""

import os

import matplotlib
from matplotlib.ft2font import FT2Font, Kerning

font = FT2Font(
    os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf'))
font.set_charmap(0)

codes = font.get_charmap().items()

# make a charname to charcode and glyphind dictionary
coded = {}
glyphd = {}
for ccode, glyphind in codes:
    name = font.get_glyph_name(glyphind)
    coded[name] = ccode
    glyphd[name] = glyphind
    # print(glyphind, ccode, hex(int(ccode)), name)

code = coded['A']
glyph = font.load_char(code)
print(glyph.bbox)
print(glyphd['A'], glyphd['V'], coded['A'], coded['V'])
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], Kerning.DEFAULT))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], Kerning.UNFITTED))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], Kerning.UNSCALED))
print('AT', font.get_kerning(glyphd['A'], glyphd['T'], Kerning.UNSCALED))
