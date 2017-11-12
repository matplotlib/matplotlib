"""
=============
Font Indexing
=============

A little example that shows how the various indexing into the font
tables relate to one another.  Mainly for Matplotlib developers...

"""

from matplotlib._ft2 import Kerning
import matplotlib.font_manager


fname = matplotlib.get_data_path() + '/fonts/ttf/DejaVuSans.ttf'
font = matplotlib.font_manager.get_font(fname)

# This assumes FreeType>=2.8.1, which automatically picks or synthesizes a
# Unicode charmap.

def get_kerning(c1, c2, mode):
    i1 = font.get_char_index(ord(c1))
    i2 = font.get_char_index(ord(c2))
    return font.get_kerning(i1, i2, mode)


print('AV default ', get_kerning('A', 'V', Kerning.DEFAULT))
print('AV unfitted', get_kerning('A', 'V', Kerning.UNFITTED))
print('AT default ', get_kerning('A', 'T', Kerning.DEFAULT))
print('AT unfitted', get_kerning('A', 'T', Kerning.UNFITTED))
