#!/usr/bin/env python
# -*- noplot -*-

from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np

files = []

fig, ax = plt.subplots(figsize=(5, 5))
for i in range(50):  # 50 frames
    plt.cla()
    plt.imshow(np.random.rand(5, 5), interpolation='nearest')
    fname = '_tmp%03d.png' % i
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)

print('Making movie animation.mpg - this make take a while')
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
#os.system("convert _tmp*.png animation.mng")

# cleanup
for fname in files:
    os.remove(fname)
