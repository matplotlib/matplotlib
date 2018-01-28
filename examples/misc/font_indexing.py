"""
=============
Font Indexing
=============

A little example that shows how the various indexing into the font
tables relate to one another.  Mainly for mpl developers....

"""
from __future__ import print_function
import matplotlib
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT, KERNING_UNFITTED, KERNING_UNSCALED


fname = matplotlib.get_data_path() + '/fonts/ttf/DejaVuSans.ttf'
font = FT2Font(fname)
font.set_charmap(0)

codes = font.get_charmap().items()
#dsu = [(ccode, glyphind) for ccode, glyphind in codes]
#dsu.sort()
#for ccode, glyphind in dsu:
#    try: name = font.get_glyph_name(glyphind)
#    except RuntimeError: pass
#    else: print('% 4d % 4d %s %s' % (glyphind, ccode, hex(int(ccode)), name))


# make a charname to charcode and glyphind dictionary
coded = {}
glyphd = {}
for ccode, glyphind in codes:
    name = font.get_glyph_name(glyphind)
    coded[name] = ccode
    glyphd[name] = glyphind

code = coded['A']
glyph = font.load_char(code)
print(glyph.bbox)
print(glyphd['A'], glyphd['V'], coded['A'], coded['V'])
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_DEFAULT))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNFITTED))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNSCALED))
print('AV', font.get_kerning(glyphd['A'], glyphd['T'], KERNING_UNSCALED))
