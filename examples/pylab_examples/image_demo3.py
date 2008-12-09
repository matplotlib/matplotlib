#!/usr/bin/env python
from pylab import *
try:
    import Image
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this example")

lena = Image.open('../data/lena.jpg')
dpi = rcParams['figure.dpi']
figsize = lena.size[0]/dpi, lena.size[1]/dpi

figure(figsize=figsize)
ax = axes([0,0,1,1], frameon=False)
ax.set_axis_off()
im = imshow(lena, origin='lower')

show()

