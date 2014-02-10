import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from  matplotlib import colors
from itertools import chain

n = len(colors.cnames)
ncols = n / 15
nrows = n / ncols

names = colors.cnames.keys()
hex_ = colors.cnames.values()
rgb = [colors.hex2color(color) for color in hex_]
hsv = [colors.rgb_to_hsv(color) for color in rgb]

hue = [color[0] for color in hsv]
sat = [color[1] for color in hsv]
val = [color[2] for color in hsv]

ind = np.lexsort((val, sat, hue))

names_ = []
colors_ = []

for i in ind:
    names_.append(names[i])
    colors_.append(hex_[i])

fig, axes_ = plt.subplots(nrows=nrows, ncols=ncols)
axes = list(chain(*axes_))

for i in range(n):
    title = axes[i].set_title(names_[i])
    title.set_size('xx-small')
    axes[i].set_axis_bgcolor(colors_[i]) 
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
