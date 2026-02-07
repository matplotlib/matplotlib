"""
==================
Radio Buttons Grid
==================

Using radio buttons in a 2D grid layout.

Radio buttons can be arranged in a 2D grid by passing a ``(rows, cols)``
tuple to the *layout* parameter. This is useful when you have multiple
related options that are best displayed in a grid format rather than a
vertical list.

In this example, we create a color picker using a 2D grid of radio buttons
to select the line color of a plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import RadioButtons

# Generate sample data
t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * t)

fig, (ax_plot, ax_buttons) = plt.subplots(
    1,
    2,
    figsize=(8, 4),
    width_ratios=[4, 1.4],
)

# Create initial plot
(line,) = ax_plot.plot(t, s, lw=2, color="red")
ax_plot.set_xlabel("Time (s)")
ax_plot.set_ylabel("Amplitude")
ax_plot.set_title("Sine Wave - Click a color!")
ax_plot.grid(True, alpha=0.3)

# Configure the radio buttons axes
ax_buttons.set_facecolor("0.9")
ax_buttons.set_title("Line Color", fontsize=12, pad=10)
# Create a 2D grid of color options (3 rows x 2 columns)
colors = ["red", "yellow", "green", "purple", "brown", "gray"]
radio = RadioButtons(ax_buttons, colors, layout=(3, 2))


def color_func(label):
    """Update the line color based on selected button."""
    line.set_color(label)
    fig.canvas.draw()


radio.on_clicked(color_func)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RadioButtons`
#
# .. tags::
#
#    styling: color
#    styling: conditional
#    plot-type: line
#    level: intermediate
#    purpose: showcase
