"""
See pcolor_demo2 for a much faster way of generating pcolor plots
"""
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


fig = plt.figure()
Z = np.arange(10000.0)
Z.shape = 100,100
Z[:,50:] = 1.

im1 = plt.figimage(Z, xo=50, yo=0, cmap=cm.jet, origin='lower')
im2 = plt.figimage(Z, xo=100, yo=100, alpha=.8, cmap=cm.jet, origin='lower')


if 0:
    dpi = 72
    plt.savefig('figimage_%d.png'%dpi, dpi=dpi, facecolor='gray')
    plt.savefig('figimage_%d.pdf'%dpi, dpi=dpi, facecolor='gray')
    plt.savefig('figimage_%d.svg'%dpi, dpi=dpi, facecolor='gray')
    plt.savefig('figimage_%d.eps'%dpi, dpi=dpi, facecolor='gray')
plt.show()



