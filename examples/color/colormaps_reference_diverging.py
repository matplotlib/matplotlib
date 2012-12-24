"""
Reference for "diverging" colormaps.

Diverging colormaps have a median value (usually light in color) and vary
smoothly to two different color tones at high and low values. Diverging
colormaps are ideal when your data has a median value that is significant (e.g.
0, such that positive and negative values are represented by different colors
of the colormap).
"""
import numpy as np
import matplotlib.pyplot as plt


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Note that any of these colormaps can be reversed by appending '_r'.
cmaps = ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy',
         'RdYlBu', 'RdYlGn', 'seismic']

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
