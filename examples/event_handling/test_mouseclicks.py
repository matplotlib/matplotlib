from __future__ import print_function

import matplotlib
#matplotlib.use("WxAgg")
#matplotlib.use("TkAgg")
#matplotlib.use("GTKAgg")
#matplotlib.use("Qt4Agg")
#matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

#print("***** TESTING WITH BACKEND: %s"%matplotlib.get_backend() + " *****")


def OnClick(event):
    if event.dblclick:
        print("DBLCLICK", event)
    else:
        print("DOWN    ", event)


def OnRelease(event):
    print("UP      ", event)


fig = plt.gcf()
cid_up = fig.canvas.mpl_connect('button_press_event', OnClick)
cid_down = fig.canvas.mpl_connect('button_release_event', OnRelease)

plt.gca().text(0.5, 0.5, "Click on the canvas to test mouse events.",
               ha="center", va="center")

plt.show()
