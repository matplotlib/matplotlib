"""
===================
Rotated tick labels
===================

Rotating tick labels can be useful when the labels are long and overlap with each other.
Adjust the tick properties using `~.Axes.tick_params`: Set the angle in degrees via
the *rotation* parameter. Set the *rotation_mode* parameter to "xtick" / "ytick" to
make the text point towards the tick, see also `~.Text.set_rotation_mode`.
"""

import matplotlib.pyplot as plt

population = {
    'India': 1417.492,
    'China': 1404.89,
    'United States': 341.784857,
    'Indonesia': 284.438782,
    'Pakistan': 241.499431,
    'Nigeria': 223.8,
    'Brazil': 213.421037,
    'Bangladesh': 169.828911,
    'Russia': 146.028325,
    'Mexico': 131.001723,
    'Japan': 122.95,
    'Philippines': 114.1236,
    'DR Congo': 112.832,
    'Ethiopia': 111.652998,
    'Egypt': 107.868296,
    'Vietnam': 102.3,
}

fig, ax = plt.subplots()
ax.bar(population.keys(), population.values())
ax.tick_params("x", rotation=45, rotation_mode="xtick")
ax.set_ylabel("population (millions)")
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.tick_params` / `matplotlib.pyplot.tick_params`
#
# .. tags:: component: ticks
