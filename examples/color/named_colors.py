import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from  matplotlib import colors

colors_ = colors.cnames.items()

#add the single letter colors
for name, rgb in colors.ColorConverter.colors.items():
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

#hex color values
hex_ = [color[1] for color in colors_]
#rgb equivalent
rgb = [colors.hex2color(color) for color in hex_]
#hsv equivalent
hsv = [colors.rgb_to_hsv(color) for color in rgb]

#split the hsv to sort
hue = [color[0] for color in hsv]
sat = [color[1] for color in hsv]
val = [color[2] for color in hsv]

#sort by hue, saturation and value
ind = np.lexsort((val, sat, hue))
sorted_colors = [colors_[i] for i in ind]

n = len(sorted_colors)
ncols = 10
nrows = int(np.ceil(1. * n / ncols))

fig = plt.figure()
for i, (name, color) in enumerate(sorted_colors):
    ax = fig.add_subplot(nrows, ncols, i + 1,
                         axisbg=color,
                         xticks=[], yticks=[])
    title = ax.set_title(name)
    title.set_size('xx-small')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()
