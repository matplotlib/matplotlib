#!/usr/bin/env python

from __future__ import print_function
from pylab import *
import matplotlib.cbook as cbook
# data are 256x256 16 bit integers
dfile = cbook.get_sample_data('s1045.ima.gz')
im = np.fromstring(dfile.read(), np.uint16).astype(float)
im.shape = 256, 256

#imshow(im, ColormapJet(256))
imshow(im, cmap=cm.gray)
axis('off')

show()
