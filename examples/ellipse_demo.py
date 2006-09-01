from pylab import *
from matplotlib.patches import Ellipse

NUM = 250

ells = [Ellipse(rand(2)*10, rand(), rand(), rand()*360) for i in xrange(NUM)]

a = subplot(111)
for e in ells:
    a.add_artist(e)
    e.set_clip_box(a.bbox)
    e.set_alpha(rand())
    e.set_facecolor(rand(3))

xlim(0, 10)
ylim(0, 10)

savefig('ellipse_demo')
show()
