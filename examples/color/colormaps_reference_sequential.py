"""
Reference for "sequential" colormaps.

Sequential colormaps are approximately monochromatic colormaps going from low
saturation (e.g. white) to high saturation (e.g. a bright blue). Sequential
colormaps are ideal for representing most scientific data since they show
a clear progression from low-to-high values.
"""
import numpy as np
import matplotlib.pyplot as plt


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Note that any of these colormaps can be reversed by appending '_r'.
cmaps = ['binary', 'Blues', 'BuGn', 'BuPu', 'gist_yarg', 'GnBu', 'Greens',
         'Greys', 'Oranges', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'Purples',
         'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']

fig, axes = plt.subplots(nrows=len(cmaps))
fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)

for ax, m in zip(axes, cmaps):
    ax.set_axis_off()
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(m))
    pos = list(ax.get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2.
    fig.text(x_text, y_text, m, va='center', ha='right')

plt.show()
