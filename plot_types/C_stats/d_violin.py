"""
=================
violinplot(D,...)
=================
"""
import matplotlib.pyplot as plt
import numpy as np

# make data:
np.random.seed(10)
D = np.random.normal((3, 5, 4), (0.75, 1.00, 0.75), (200, 3))

# plot:
with plt.style.context('cheatsheet_gallery'):
    fig, ax = plt.subplots()

    VP = ax.violinplot(D, [2, 4, 6], widths=2,
                       showmeans=False, showmedians=False, showextrema=False)
    #style:
    for body in VP['bodies']:
        body.set_alpha(0.9)

ax.set_xlim(0, 8)
ax.set_xticks(np.arange(1, 8))
ax.set_ylim(0, 8)
ax.set_yticks(np.arange(1, 8))

plt.show()
