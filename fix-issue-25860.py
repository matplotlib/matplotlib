import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np

def on_pick(evt):
    print('pick event!')

fig = plt.figure()
gs = mgridspec.GridSpec(1, 1)

subfig = fig.add_subplot(gs[:, 0])

subfig.plot(np.random.random(10), np.random.random(10), marker='.', linestyle='none', picker=5)

fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()