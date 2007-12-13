#!/usr/bin/env python
from pylab import *

# data are 256x256 16 bit integers
dfile = 'data/s1045.ima'
im = fromstring(file(dfile, 'rb').read(), uint16).astype(float)
im.shape = 256, 256

#imshow(im, ColormapJet(256))
imshow(im, cmap=cm.jet)
axis('off')
#savefig('mri_demo')
show()
