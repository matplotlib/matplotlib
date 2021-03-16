"""
=================
eventplot(D, ...)
=================
"""
import matplotlib.pyplot as plt
import numpy as np

# make data:
np.random.seed(1)
X = [2, 4, 6]
D = np.random.gamma(4, size=(3, 50))

# plot:
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    ax.eventplot(D, colors="C0", orientation="vertical",
                 lineoffsets=X, linewidth=0.75)

    ax.set_xlim(0, 8)
    ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(0, 8)
    ax.set_yticks(np.arange(1, 8))

plt.show()
