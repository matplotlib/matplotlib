#issue https://github.com/matplotlib/matplotlib/issues/2164

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import cnames
from itertools import chain

print len(cnames.keys())
n = len(cnames)
ncols = n / 15
nrows = n / ncols
names = cnames.keys()
colors = cnames.values()

fig, axes_ = plt.subplots(nrows=nrows, ncols=ncols)
axes = list(chain(*axes_))

for i in range(n):
    title = axes[i].set_title(names[i])
    title.set_size('xx-small')
    axes[i].set_axis_bgcolor(colors[i]) 
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

#     axes[i].set_label(names[i])

# for axes in list(chain(*axes_)):
#     axes.set_axis_off()

# print nrows, ncols, fig, axes
plt.show()