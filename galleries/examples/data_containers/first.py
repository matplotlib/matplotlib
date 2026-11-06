"""
=================
A functional line
=================

Demonstrating the differences between :class:`.containers.FuncContainer` and
:class:`.containers.ArrayContainer` using :class:`Line2D`.

Initially empty lines are created, and then their containers are attached,
using :meth:`.Line2D.set_container`.

An ArrayContainer has static data that does not update.
A FuncContainer will recompute according to the view window of the Axes.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.data_containers.containers import ArrayContainer, FuncContainer

fc = FuncContainer(
    xfuncs={"x": (("N",), lambda x: x), "y": (("N",), lambda x: np.sin(1 / x))}
)
fc2 = FuncContainer(
    xfuncs={"x": (("N",), lambda x: x), "y": (("N",), lambda x: np.cos(x))}
)

th = np.linspace(0, 2 * np.pi, 16)
ac = ArrayContainer(x=th, y=np.cos(th))

fig, ax = plt.subplots()
line1, line2, line3, *_ = ax.plot([], [], "-", [], [], "--", [], [], "o")

line1.set_container(fc)
line2.set_container(fc2)
line3.set_container(ac)


ax.set_xlim(0, np.pi * 4)
ax.set_ylim(-1.1, 1.1)

plt.show()
