"""
===================
Rotated tick labels
===================
"""

import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1, 4, 9, 6]
labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']

fig, ax = plt.subplots()
ax.plot(x, y)
# A tick label rotation can be set using Axes.tick_params.
ax.tick_params("y", rotation=45)
# Alternatively, if setting custom labels with set_xticks/set_yticks, it can
# be set at the same time as the labels.
# For both APIs, the rotation can be an angle in degrees, or one of the strings
# "horizontal" or "vertical".
ax.set_xticks(x, labels, rotation='vertical')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.tick_params` / `matplotlib.pyplot.tick_params`
#    - `matplotlib.axes.Axes.set_xticks` / `matplotlib.pyplot.xticks`
