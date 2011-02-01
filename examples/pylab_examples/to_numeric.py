#!/usr/bin/env python
"""
Use backend agg to access the figure canvas as an RGB string and then
convert it to an array and pass the string it to PIL for
rendering
"""

from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from PIL import Image
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this example")

plot([1,2,3])

canvas = get_current_fig_manager().canvas

agg = canvas.switch_backends(FigureCanvasAgg)
agg.draw()
s = agg.tostring_rgb()

# get the width and the height to resize the matrix
l,b,w,h = agg.figure.bbox.bounds
w, h = int(w), int(h)


X = fromstring(s, uint8)
X.shape = h, w, 3

im = Image.fromstring( "RGB", (w,h), s)
# im.show()

