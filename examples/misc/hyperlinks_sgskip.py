"""
==========
Hyperlinks
==========

This example demonstrates how to set a hyperlinks on various kinds of elements.

This currently only works with the SVG backend.

"""


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

###############################################################################

f = plt.figure()
s = plt.scatter([1, 2, 3], [4, 5, 6])
s.set_urls(['http://www.bbc.co.uk/news', 'http://www.google.com', None])
f.savefig('scatter.svg')

###############################################################################

f = plt.figure()
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                origin='lower', extent=[-3, 3, -3, 3])

im.set_url('http://www.google.com')
f.savefig('image.svg')
