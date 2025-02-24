"""
======================
Broken horizontal bars
======================

`~.Axes.broken_barh` creates sequences of horizontal bars. This example shows
a timing diagram.
"""
import matplotlib.pyplot as plt
import numpy as np

# data is a sequence of (start, duration) tuples
cpu_1 = [(0, 3), (3.5, 1), (5, 5)]
cpu_2 = np.column_stack([np.linspace(0, 9, 10), np.full(10, 0.5)])
cpu_3 = np.column_stack([10*np.random.random(61), np.full(61, 0.05)])
cpu_4 = [(2, 1.7), (7, 1.2)]
disk = [(1, 1.5)]
network = np.column_stack([10*np.random.random(10), np.full(10, 0.05)])

fig, ax = plt.subplots()
# broken_barh(xranges, (ymin, height))
ax.broken_barh(cpu_1, (-0.2, 0.4))
ax.broken_barh(cpu_2, (0.8, 0.4))
ax.broken_barh(cpu_3, (1.8, 0.4))
ax.broken_barh(cpu_4, (2.8, 0.4))
ax.broken_barh(disk, (3.8, 0.4), color="tab:orange")
ax.broken_barh(network, (4.8, 0.4), color="tab:green")
ax.set_xlim(0, 10)
ax.set_yticks(range(6),
              labels=["CPU 1", "CPU 2", "CPU 3", "CPU 4", "disk", "network"])
ax.invert_yaxis()
ax.set_title("Resource usage")

plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.broken_barh` / `matplotlib.pyplot.broken_barh`
#    - `matplotlib.axes.Axes.invert_yaxis`
#    - `matplotlib.axes.Axes.set_yticks`
#
# .. tags::
#
#    component: annotation
#    plot-type: bar
#    level: beginner
