#!/usr/bin/env python
from matplotlib.matlab import *

w, h = 512, 512
s = file('data/ct.raw', 'rb').read()
A = fromstring(s, UInt16).astype(Float)
A *= 1.0/max(A)
A.shape = w, h

im = imshow(A, cmap=cm.jet, origin='upper')
#im.set_aspect('preserve')

# plot some data with the image; currently broken with aspect preserve
gca().set_image_extent(0, 25, 0, 25)
markers = [(15.9, 14.5), (16.8, 15)]
x,y = zip(*markers)
plot(x, y, 'o')
#axis([0,25,0,25])



#axis('off')
title('CT density')

if 0:
    x = sum(A,0)
    subplot(212)
    bar(arange(w), x)
    set(gca(), 'xlim', [0,h-1])
    ylabel('density')
    set(gca(), 'xticklabels', [])

#savefig('image_demo2')
show()

