import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

fig, ax = plt.subplots()
ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
ax.plot(t, s1, lw=2, color='r', label='4 Hz')
ax.plot(t, s2, lw=2, color='g', label='6 Hz')
plt.subplots_adjust(left=0.2)

lines = ax.get_lines()

# Make checkbuttons with all plotted lines with correct visibility
rax = plt.axes([0.05, 0.4, 0.1, 0.15])
labels = [str(graph.get_label()) for graph in lines]
visibility = [graph.get_visible() for graph in lines]
check = CheckButtons(rax, labels, visibility)

def func(label):
    lines[labels.index(label)].set_visible(not lines[labels.index(label)].get_visible())
    plt.draw()

check.on_clicked(func)

plt.show()
