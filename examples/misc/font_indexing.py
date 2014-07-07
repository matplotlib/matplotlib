"""
A little example that shows how the various indexing into the font
tables relate to one another.  Mainly for mpl developers....

"""
from __future__ import print_function
import matplotlib
import matplotlib.ft2font as ft


# fname = '/usr/share/fonts/sfd/FreeSans.ttf'
fname = matplotlib.get_data_path() + '/fonts/ttf/Vera.ttf'
font = ft.FT2Font(fname)
font.set_charmap(0)

codes = font.get_charmap().items()
# dsu = [(ccode, glyphind) for ccode, glyphind in codes]
# dsu.sort()
# for ccode, glyphind in dsu:
#    try: name = font.get_glyph_name(glyphind)
#    except RuntimeError: pass
#    else: print '% 4d % 4d %s %s'%(glyphind, ccode, hex(int(ccode)), name)


# make a charname to charcode and glyphind dictionary
coded = {}
glyphd = {}
for ccode, glyphind in codes:
    name = font.get_glyph_name(glyphind)
    coded[name] = ccode
    glyphd[name] = glyphind

code = coded['A']
glyph = font.load_char(code)
# print glyph.bbox
print(glyphd['A'], glyphd['V'], coded['A'], coded['V'])
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], ft.KERNING_DEFAULT))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], ft.KERNING_UNFITTED))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], ft.KERNING_UNSCALED))
print('AV', font.get_kerning(glyphd['A'], glyphd['T'], ft.KERNING_UNSCALED))
