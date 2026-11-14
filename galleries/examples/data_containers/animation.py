"""
================
An animated line
================

An animated line using a custom container class.

This is a demonstration of a completely custom, duck typed, Container class.

While this particular example could likely be accomplished with
:class:`.FuncContainer`, this is an example of what is possible.

The Animation framework is used, but only to prompt new draws, the update
function for the animation is a no-op.
"""

from functools import partial
import time
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.data_containers.conversion_edge import Graph
from matplotlib.data_containers.description import Desc


class SinOfTime:
    N = 1024
    # cycles per minutes
    scale = 10

    def describe(self):
        return {
            "x": Desc((self.N,)),
            "y": Desc((self.N,)),
            "phase": Desc(()),
            "time": Desc(()),
        }

    def query(
        self,
        graph: Graph,
        parent_coordinates: str = "axes",
    ) -> tuple[dict[str, Any], Union[str, int]]:
        th = np.linspace(0, 2 * np.pi, self.N)

        cur_time = time.time()

        phase = 2 * np.pi * (self.scale * cur_time % 60) / 60
        return {
            "x": th,
            "y": np.sin(th + phase),
        }, hash(cur_time)


def update(frame, art):
    return art


sot_c = SinOfTime()

fig, ax = plt.subplots()

# Initially plot empty, then set the container
line, *_ = ax.plot([], [])
line.set_container(sot_c)

ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.1, 1.1)

# This is a bit of a kludge to use animation to automate redrawing
ani = FuncAnimation(
    fig,
    partial(update, art=(line,)),
    frames=25,
    interval=1000 / 60,
)

plt.show()
