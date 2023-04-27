"""
============
Ellipse with arrow Demo
============

Draw an ellipses with an arrow showing rotation direction. Compare this
to the :doc:`Ellipse collection example
</gallery/shapes_and_collections/ellipse_collection>`.
"""

import matplotlib.pyplot as plt
import numpy as np

# Define start position of ellipse
xVec = 0.5 + 0.5j
yVec = 0.2 + 0.5j

sampling = 101
n = np.linspace(0, sampling, sampling)

# Calculate ellipse data points
x = np.real(xVec * np.exp(1j * 2 * np.pi * n / sampling))
y = np.real(yVec * np.exp(1j * 2 * np.pi * n / sampling))

# Draw ellipse
fig, ax = plt.subplots(1, 1, subplot_kw={"aspect": "equal"})
ax.plot(x, y)

# Calculate arrow position and orientation
dx = x[-1] - x[-2]
dy = y[-1] - y[-2]
ax.arrow(x=x[-1], y=y[-1], dx=dx, dy=dy, head_width=0.05)

ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Ellipse`
