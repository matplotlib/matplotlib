#!/usr/bin/env python

from pylab import *

fig = figure()
subplot(221)
imshow(rand(100,100))
subplot(222)
imshow(rand(100,100))
subplot(223)
imshow(rand(100,100))
subplot(224)
imshow(rand(100,100))

subplot_tool()
show()
