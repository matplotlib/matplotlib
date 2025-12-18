"""
==============================
Text and mathtext using pyplot
==============================

Set the special text objects `~.pyplot.title`, `~.pyplot.xlabel`, and
`~.pyplot.ylabel` through the dedicated pyplot functions.  Additional text
objects can be placed in the Axes using `~.pyplot.text`.

You can use TeX-like mathematical typesetting in all texts; see also
:ref:`mathtext`.

.. redirect-from:: /gallery/pyplots/pyplot_mathtext
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)

plt.plot(t, s)
plt.text(0, -1, r'Hello, world!', fontsize=15)
plt.title(r'$\mathcal{A}\sin(\omega t)$', fontsize=20)
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
#    - `matplotlib.pyplot.hist`
#    - `matplotlib.pyplot.xlabel`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.text`
#    - `matplotlib.pyplot.grid`
#    - `matplotlib.pyplot.show`
