from matplotlib.matlab import *

w, h = 512, 512
s = file('data/ct.raw', 'rb').read()
A = fromstring(s, typecode=UInt16).astype(Float)
A *= 1.0/max(A)
A.shape = w, h

markers = [(15.9, 14.5), (16.8, 15)]
x,y = zip(*markers)
#figure(1, figsize=(2,7))
#pyt subplot(211)
im = imshow(A)
im.set_datalimx(0,25)
im.set_datalimy(0,25)
#im.set_interpolation('bicubic')
#im.set_interpolation('nearest')
im.set_interpolation('bilinear')
#im.set_aspect('free')
im.set_aspect('preserve')
print x, y
plot(x, y, 'o')
axis([0,25,0,25])



#axis('off')
title('CT density')

if 0:
    x = sum(A,0)
    subplot(212)
    bar(arange(w), x)
    set(gca(), 'xlim', [0,h-1])
    ylabel('density')
    set(gca(), 'xticklabels', [])
show()

