"""
===============
Scales overview
===============

Illustrate the scale transformations applied to axes, e.g. log, symlog, logit.

See `matplotlib.scale` for a full list of built-in scales, and
:doc:`/gallery/scales/custom_scale` for how to create your own scale.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(400)
y = np.linspace(0.002, 1, 400)

fig, axs = plt.subplots(3, 2, figsize=(6, 8), layout='constrained')

axs[0, 0].plot(x, y)
axs[0, 0].set_yscale('linear')
axs[0, 0].set_title('linear')
axs[0, 0].grid(True)

axs[0, 1].plot(x, y)
axs[0, 1].set_yscale('log')
axs[0, 1].set_title('log')
axs[0, 1].grid(True)

axs[1, 0].plot(x, y - y.mean())
axs[1, 0].set_yscale('symlog', linthresh=0.02)
axs[1, 0].set_title('symlog')
axs[1, 0].grid(True)

axs[1, 1].plot(x, y)
axs[1, 1].set_yscale('logit')
axs[1, 1].set_title('logit')
axs[1, 1].grid(True)

axs[2, 0].plot(x, y - y.mean())
axs[2, 0].set_yscale('asinh', linear_width=0.01)
axs[2, 0].set_title('asinh')
axs[2, 0].grid(True)


# Function x**(1/2)
def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2


axs[2, 1].plot(x, y)
axs[2, 1].set_yscale('function', functions=(forward, inverse))
axs[2, 1].set_title('function: $x^{1/2}$')
axs[2, 1].grid(True)
axs[2, 1].set_yticks(np.arange(0, 1.2, 0.2))

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.set_xscale`
#    - `matplotlib.axes.Axes.set_yscale`
#    - `matplotlib.scale.LinearScale`
#    - `matplotlib.scale.LogScale`
#    - `matplotlib.scale.SymmetricalLogScale`
#    - `matplotlib.scale.LogitScale`
#    - `matplotlib.scale.FuncScale`
