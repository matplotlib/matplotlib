import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y)

hline = ax.axhline(color="gray")
vline = ax.axvline(color="gray")

background = None

def on_move(event):
    global background
    if not event.inaxes:
        return

    canvas = fig.canvas

    if background is None:
        background = canvas.copy_from_bbox(ax.bbox)

    canvas.restore_region(background)

    hline.set_ydata(event.ydata)
    vline.set_xdata(event.xdata)

    ax.draw_artist(hline)
    ax.draw_artist(vline)

    canvas.blit(ax.bbox)

fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()