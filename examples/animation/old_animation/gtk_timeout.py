import gobject
import numpy as np
import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0, 1)

def update():
    line.set_ydata(np.random.rand(10))
    fig.canvas.draw_idle()
    return True  # return False to terminate the updates

gobject.timeout_add(100, update)  # you can also use idle_add to update when gtk is idle
plt.show()
