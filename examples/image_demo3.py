#!/usr/bin/env python
from matplotlib.matlab import *
import Image

lena = Image.open('data/lena.jpg')
dpi = rcParams['figure.dpi']
figsize = lena.size[0]/dpi, lena.size[1]/dpi

figure(figsize=figsize)

im = imshow(lena, origin='lower', aspect='preserve')

#savefig('image_demo3')
show()

