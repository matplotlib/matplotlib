from matplotlib.image import ASPECT_FREE, ASPECT_PRESERVE
from matplotlib.image import NEAREST, BILINEAR, BICUBIC, SPLINE16
from matplotlib.matlab import *


s = file('data/ct.raw', 'rb').read()
A = fromstring(s, typecode=UInt16).astype(Float)
A *= 1.0/max(A)
A.shape = 512, 512

im = imshow(A)
#im.set_interpolation(BICUBIC)
#im.set_interpolation(NEAREST)
im.set_interpolation(BILINEAR)
#im.set_preserve_aspect(ASPECT_PRESERVE)



show()

