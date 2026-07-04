"""
=============
Font indexing
=============

This example shows how the font tables relate to one another.
"""

# sphinx_gallery_thumbnail_path = "_static/font_indexing.png"
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


# A mapping of characters to what they are
# caron = "a letter v-shaped sign placed over a letter to indicate a change of "
#         "pronunciation"


chars = {
    'LATIN SMALL LETTER A WITH DIAERESIS': ('\u00e4', 'A with umlaut'),
    'LATIN SMALL LETTER A WITH MACRON': ('\u0101', 'A with macron'),
    'LATIN SMALL LETTER A WITH CIRCUMFLEX': ('\u00e2', 'A with circumflex'),
    'LATIN SMALL LETTER A WITH CARON': ('\u01ce', 'A with caron'),
}

# iterate over the font glyphs
for long_name, (char, short) in chars.items():
    try:
        code = coded[long_name]
    except KeyError:
        continue
    glyph = font.load_char(
        code,
        flags=matplotlib.ft2font.LOAD_NO_HINTING |
              matplotlib.ft2font.LOAD_NO_BITMAP)
    print(f'{short}\n   glyph {glyphd[long_name]}, char code {code}')
    print(f'   {glyph.bbox}')
    print(f'   advanced x: {glyph.linear_vert_advance}')
    print()
