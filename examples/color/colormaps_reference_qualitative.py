"""
Reference for "qualitative" colormaps.

Qualitative colormaps vary rapidly in color. These colormaps are useful for
choosing a set of discrete colors. For example::

    color_list = plt.cm.Set3(np.linspace(0, 1, 12))

gives a list of RGB colors that are good for plotting a series of lines on
a dark background.
"""
import numpy as np
import matplotlib.pyplot as plt


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Note that any of these colormaps can be reversed by appending '_r'.
cmaps = ['Accent', 'Dark2', 'hsv', 'Paired', 'Pastel1', 'Pastel2', 'Set1',
         'Set2', 'Set3', 'spectral']

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
