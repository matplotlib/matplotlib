from __future__ import print_function
import time, sys
import numpy as np
import matplotlib.pyplot as plt


def data_gen():
    t = data_gen.t
    data_gen.t += 0.05
    return np.sin(2*np.pi*t) * np.exp(-t/10.)
data_gen.t = 0

fig, ax = plt.subplots()
line, = ax.plot([], [], animated=True, lw=2)
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 5)
ax.grid()
xdata, ydata = [], []
def run(*args):
    background = fig.canvas.copy_from_bbox(ax.bbox)
    # for profiling
    tstart = time.time()

    while 1:
        # restore the clean slate background
        fig.canvas.restore_region(background)
        # update the data
        t = data_gen.t
        y = data_gen()
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()
        if t>=xmax:
            ax.set_xlim(xmin, 2*xmax)
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(ax.bbox)

        line.set_data(xdata, ydata)

        # just draw the animated artist
        ax.draw_artist(line)
        # just redraw the axes rectangle
        fig.canvas.blit(ax.bbox)

        if run.cnt==1000:
            # print the timing info and quit
            print('FPS:' , 1000/(time.time()-tstart))
            sys.exit()

        run.cnt += 1
run.cnt = 0



manager = plt.get_current_fig_manager()
manager.window.after(100, run)

plt.show()
