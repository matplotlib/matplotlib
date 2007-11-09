#!/usr/bin/env python
from pylab import *

w, h = 512, 512
s = file('data/ct.raw', 'rb').read()
A = fromstring(s, uint16).astype(float)
A *= 1.0/max(A)
A.shape = w, h

extent = (0, 25, 0, 25)
im = imshow(A, cmap=cm.jet, origin='upper', extent=extent)

markers = [(15.9, 14.5), (16.8, 15)]
x,y = zip(*markers)
plot(x, y, 'o')
#axis([0,25,0,25])



#axis('off')
title('CT density')

if 0:
    x = asum(A,0)
    subplot(212)
    bar(arange(w), x)
    xlim(0,h-1)
    ylabel('density')
    setp(gca(), 'xticklabels', [])

#savefig('image_demo2')
show()

