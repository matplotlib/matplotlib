#!/usr/bin/env python
from pylab import *
try:
    import Image
except ImportError, exc:
    raise SystemExit("PIL must be loaded to run this example")

lena = Image.open('data/lena.jpg')
dpi = rcParams['figure.dpi']
figsize = lena.size[0]/dpi, lena.size[1]/dpi

figure(figsize=figsize)

im = imshow(lena, origin='lower')

#savefig('image_demo3')
show()

