import sys, time, os
from matplotlib.numerix import rand
from matplotlib.ft2font import FT2Font
from matplotlib.backends.backend_ps import encodeTTFasPS

fname = '/usr/local/share/matplotlib/Vera.ttf'

def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[0])

fname = '/usr/local/share/matplotlib/Vera.ttf'
N = 400
for i in range(N):
    font = FT2Font(fname)
    font.clear()
    font.set_text('hi mom', 60)
    font.set_size(12, 72)

    #glyph = font.load_char(int(140*rand()))
    font.draw_glyphs_to_bitmap()
    #font.draw_glyph_to_bitmap(0, 0, glyph)


    #s = encodeTTFasPS(fname)
    val = report_memory(i)
    if i==1: start = val

end = val
print 'Average memory consumed per loop: %1.4f\n' % ((end-start)/float(N))

# Average memory consumed per loop: 0.09
