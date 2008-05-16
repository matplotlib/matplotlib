import numpy as np
from numpy import ma
from matplotlib.pyplot import step, legend, xlim, ylim, show

x = np.arange(1, 7, 0.4)
y0 = np.sin(x)
y = y0.copy() + 2.5

step(x, y, label='pre (default)')

y -= 0.5
step(x, y, where='mid', label='mid')

y -= 0.5
step(x, y, where='post', label='post')

y = ma.masked_where((y0>-0.15)&(y0<0.15), y - 0.5)
step(x,y, label='masked (pre)')

legend()

xlim(0, 7)
ylim(-0.5, 4)

show()

