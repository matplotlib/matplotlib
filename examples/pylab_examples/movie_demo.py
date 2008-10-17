#!/usr/bin/env python
# -*- noplot -*-

import os, sys
from pylab import *

files = []
figure(figsize=(5,5))
ax = subplot(111)
for i in range(50):  # 50 frames
    cla()
    imshow(rand(5,5), interpolation='nearest')
    fname = '_tmp%03d.png'%i
    print 'Saving frame', fname
    savefig(fname)
    files.append(fname)

print 'Making movie animation.mpg - this make take a while'
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
#os.system("convert _tmp*.png animation.mng")

# cleanup
for fname in files: os.remove(fname)
