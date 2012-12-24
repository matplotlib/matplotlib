"""
Reference for "sequential" colormaps.

These are sequential colormaps, which vary smoothly between two color tones.
Unlike the sequential colormaps in the first example, these either vary between
two saturated tones (instead of being nearly monochromatic) or have a reversed
progression (decreasing in saturation for increasing value). Note, however,
that any colormap can be reversed by appending "_r" (e.g., "pink_r").
"""
import numpy as np
import matplotlib.pyplot as plt


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Note that any of these colormaps can be reversed by appending '_r'.
cmaps = ['afmhot', 'autumn', 'bone', 'cool', 'copper', 'gist_gray',
         'gist_heat', 'gray', 'hot', 'pink', 'spring', 'summer', 'winter']

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
