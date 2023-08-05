"""
===============
Watermark image
===============

Overlay an image on a plot by moving it to the front (``zorder=3``) and making it
semi-transparent (``alpha=0.7``).
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook
import matplotlib.image as image

with cbook.get_sample_data('logo2.png') as file:
    im = image.imread(file)

fig, ax = plt.subplots()

np.random.seed(19680801)
x = np.arange(30)
y = x + np.random.randn(30)
ax.bar(x, y, color='#6bbc6b')
ax.grid()

fig.figimage(im, 25, 25, zorder=3, alpha=.7)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.image`
#    - `matplotlib.image.imread` / `matplotlib.pyplot.imread`
#    - `matplotlib.figure.Figure.figimage`
