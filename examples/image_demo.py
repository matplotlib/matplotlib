from matplotlib.image import ASPECT_FREE, ASPECT_PRESERVE
from matplotlib.image import NEAREST, BILINEAR, BICUBIC, SPLINE16
from matplotlib.matlab import *

w, h = 512, 512
s = file('data/ct.raw', 'rb').read()
A = fromstring(s, typecode=UInt16).astype(Float)
A *= 1.0/max(A)
A.shape = w, h


figure(1, figsize=(2,7))
subplot(211)
im = imshow(A)
#im.set_interpolation(BICUBIC)
#im.set_interpolation(NEAREST)
im.set_interpolation(BILINEAR)
#im.set_preserve_aspect(ASPECT_PRESERVE)
set(gca(), 'xlim', [0,h-1])
axis('off')
title('CT density')

x = sum(A,0)
subplot(212)
bar(arange(w), x)
set(gca(), 'xlim', [0,h-1])
ylabel('density')
set(gca(), 'xticklabels', [])
show()

