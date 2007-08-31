import sys, time, os
from matplotlib.ft2font import FT2Font
from matplotlib.numerix import rand
from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.backends._backend_agg import RendererAgg

def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[0])

fname = '/usr/local/share/matplotlib/Vera.ttf'

N = 200
for i in range(N):
    gc = GraphicsContextBase()
    gc.set_clip_rectangle( [20,20,20,20] )
    o = RendererAgg(400,400, 72)

    for j in range(50):
        xs = [400*int(rand()) for k in range(8)]
        ys = [400*int(rand()) for k in range(8)]
        rgb = (1,0,0)
        pnts = zip(xs, ys)
        o.draw_polygon(gc, rgb, pnts)
        o.draw_polygon(gc, None, pnts)

    for j in range(50):
        x = [400*int(rand()) for k in range(4)]
        y = [400*int(rand()) for k in range(4)]
        o.draw_lines( gc, x, y)

    for j in range(50):
        args = [400*int(rand()) for k in range(4)]
        rgb = (1,0,0)
        o.draw_rectangle(gc, rgb, *args)

    if 1: # add text
        font = FT2Font(fname)
        font.clear()
        font.set_text('hi mom', 60)
        font.set_size(12, 72)
        o.draw_text_image(font.get_image(), 30, 40, gc)

    o.write_png('aggtest%d.png'%i)
    val = report_memory(i)
    if i==1: start = val

end = val
print 'Average memory consumed per loop: %1.4f\n' % ((end-start)/float(N))

# w/o text and w/o write_png: Average memory consumed per loop: 0.02
# w/o text and w/ write_png : Average memory consumed per loop: 0.3400
# w/ text and w/ write_png  : Average memory consumed per loop: 0.32
