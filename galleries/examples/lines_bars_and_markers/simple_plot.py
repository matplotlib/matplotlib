"""
=========
Line plot
=========

Create a basic line plot.
"""

import matplotlib.pyplot as plt
import numpy as np


# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# Creating the figure and axis
fig, ax = plt.subplots()

# Improvement: Explicitly setting line properties for better visibility
ax.plot(t, s, color='tab:blue', linewidth=2.5, label='Sine wave')

# Improvement: Standardizing label formatting and adding a legend
ax.set(xlabel='Time (s)', ylabel='Voltage (mV)',title='Basic Sine Wave Plot')

# Improvement: Adding a styled grid for easier data reading
# ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend()

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.figure.Figure.savefig`
#
# .. tags::
#
#    plot-type: line
#    level: beginner

# %%

