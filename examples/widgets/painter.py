"""
===============================
Painter Tool Demo
===============================

Drag the mouse to paint selected areas of the plot. The callback prints
the (x, y) coordinates of the center of the painted region.

"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Painter

# Create a figure and axes for the plot.
fig, ax = plt.subplots()


# Generate data and create a scatter plot.
data = np.random.rand(100, 2)
data[:, 1] *= 2
pts = ax.scatter(data[:, 0], data[:, 1], s=80)


# Define the "on_select" callback.
def callback(x, y):
    print("(%3.2f, %3.2f)" % (x, y))


print("\n click and drag \n (x, y)")

# Create the painter tool and show the plot.
p = Painter(ax, callback)
p.label = 1  # set the color
plt.show()
