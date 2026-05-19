"""
==========
Basic plot
==========

A basic plot using the :ref:`pyplot_interface`.

- `~.pyplot.plot` plots the data y versus x as lines and/or markers.
- `~.pyplot.title`, `~.pyplot.xlabel` and `~.pyplot.ylabel` set the title,
  x-axis label and y-axis label.
- `~.pyplot.show` displays the plot.

.. redirect-from:: /gallery/pyplots/fig_axes_labels_simple
.. redirect-from:: /gallery/pyplots/pyplot_formatstr
.. redirect-from:: /gallery/pyplots/pyplot_text
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2 * np.pi * x)

plt.plot(x, y)
plt.title("A basic plot using pyplot")
plt.xlabel('Time [s]')
plt.ylabel('Voltage [mV]')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.title`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.show`
