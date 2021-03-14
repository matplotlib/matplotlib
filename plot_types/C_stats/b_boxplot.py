"""
==============
boxplot(X,...)
==============
"""
import matplotlib.pyplot as plt
import numpy as np

# make data:
np.random.seed(10)
D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))

# plot
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    VP = ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white",
                                 "linewidth": 0.5},
                    boxprops={"facecolor": "C1",
                              "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C1",
                                  "linewidth": 1.5},
                    capprops={"color": "C1",
                              "linewidth": 1.5})

    ax.set_xlim(0, 8), ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(0, 8), ax.set_yticks(np.arange(1, 8))

plt.show()
