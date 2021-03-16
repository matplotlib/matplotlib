"""
====================
pie(X, explode, ...)
====================
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# plot
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    # make data
    X = [1, 2, 3, 4]
    colors = np.zeros((len(X), 4))
    colors[:] = mpl.colors.to_rgba("C0")
    colors[:, 3] = np.linspace(0.25, 0.75, len(X))

    ax.pie(X, colors=["white"]*len(X), radius=3, center=(4, 4),
            wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    ax.pie(X, colors=colors, radius=3, center=(4, 4),
            wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

    ax.set_xlim(0, 8)
    ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(0, 8)
    ax.set_yticks(np.arange(1, 8))

    plt.show()
