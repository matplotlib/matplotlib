"""
=============
Radio Buttons
=============

Using radio buttons to choose properties of your plot.

Radio buttons let you choose between multiple options in a visualization.
In this case, the buttons let the user choose one of the three different sine
waves to be shown in the plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(8*np.pi*t)

fig, ax = plt.subplots()
l, = ax.plot(t, s0, lw=2, color='red')
fig.subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('1 Hz', '2 Hz', '4 Hz'),
                     label_props={'color': 'cmy', 'fontsize': [12, 14, 16]},
                     radio_props={'s': [16, 32, 64]})


def hzfunc(label):
    hzdict = {'1 Hz': s0, '2 Hz': s1, '4 Hz': s2}
    ydata = hzdict[label]
    l.set_ydata(ydata)
    fig.canvas.draw()
radio.on_clicked(hzfunc)

rax = fig.add_axes([0.05, 0.4, 0.15, 0.15], facecolor=axcolor)
radio2 = RadioButtons(
    rax, ('red', 'blue', 'green'),
    label_props={'color': ['red', 'blue', 'green']},
    radio_props={
        'facecolor': ['red', 'blue', 'green'],
        'edgecolor': ['darkred', 'darkblue', 'darkgreen'],
    })


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw()
radio2.on_clicked(colorfunc)

rax = fig.add_axes([0.05, 0.1, 0.15, 0.15], facecolor=axcolor)
radio3 = RadioButtons(rax, ('-', '--', '-.', ':'))


def stylefunc(label):
    l.set_linestyle(label)
    fig.canvas.draw()
radio3.on_clicked(stylefunc)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RadioButtons`
