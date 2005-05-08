"""
A little example that shows how the various indexing into the font
tables relate to one another.  Mainly for mpl developers....

"""
import matplotlib
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT



fname = matplotlib.get_data_path() + '/cmr10.ttf'
font = FT2Font(fname)

codes = font.get_charmap().items()
dsu = [(ccode, glyphind) for glyphind, ccode in codes]
dsu.sort()
for ccode, glyphind in dsu: 
    try: name = font.get_glyph_name(glyphind)
    except RuntimeError: pass
    else: print '% 4d % 4d %s %s'%(glyphind, ccode, hex(int(ccode)), name)



# make a charname to charcode and glyphind dictionary
coded = {}
glyphd = {}
for glyphind, ccode in codes:
    name = font.get_glyph_name(glyphind)
    coded[name] = ccode
    glyphd[name] = glyphind

code =  coded['A']
glyph = font.load_char(code)
print glyph.bbox

print 'AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_DEFAULT)/64.0
print 'AA', font.get_kerning(glyphd['A'], glyphd['A'], KERNING_DEFAULT)/64.0
