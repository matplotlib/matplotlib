from matplotlib.matlab import *

w, h = 512, 512
s = file('data/ct.raw', 'rb').read()
A = fromstring(s, typecode=UInt16).astype(Float)
A *= 1.0/max(A)
A.shape = w,h
im = imshow(A)

# set the interpolation method: 'nearerst', 'bilinear', 'bicubic' and much more
im.set_interpolation('bilinear')

# aspect ratio 'free' or 'preserve'
im.set_aspect('preserve')

axis('off')
title('CT density')
show()

