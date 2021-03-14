"""
======================
plot([X], Y, [fmt]...)
======================
"""

import matplotlib.pyplot as plt
import numpy as np

# make data
X = np.linspace(0, 10, 100)
Y = 4 + 2 * np.sin(2 * X)

# plot
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    ax.plot(X, Y, color="C1", linewidth=2.0)

    ax.set_xlim(0, 8), ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(0, 8), ax.set_yticks(np.arange(1, 8))

    plt.show()
