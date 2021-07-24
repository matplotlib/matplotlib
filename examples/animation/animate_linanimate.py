import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import linanimate

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-')


def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    xdata.append(frame)
    ydata.append(np.cos(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani, fps = linanimate(fig, update,  init_func=init, lf=0,
                      uf=2*np.pi, fps=60, duration=5)
ani.save('linanimate_example.mp4', fps=fps)
