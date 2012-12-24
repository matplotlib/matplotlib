"""
Reference for "miscellaneous" colormaps.
"""
import numpy as np
import matplotlib.pyplot as plt


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Note that any of these colormaps can be reversed by appending '_r'.
cmaps = ['flag', 'gist_earth', 'gist_ncar', 'gist_rainbow', 'gist_stern',
         'jet', 'prism', 'brg', 'CMRmap', 'cubehelix', 'gnuplot', 'gnuplot2',
         'ocean', 'rainbow', 'terrain']

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
