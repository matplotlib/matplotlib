"""
================================
Annotation arrow style reference
================================

Overview of the arrow styles available in `~.Axes.annotate`.
"""

import inspect
import itertools
import re

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

styles = mpatches.ArrowStyle.get_styles()
ncol = 2
nrow = (len(styles) + 1) // ncol
axs = (plt.figure(figsize=(4 * ncol, 1 + nrow))
       .add_gridspec(1 + nrow, ncol,
                     wspace=0, hspace=0, left=0, right=1, bottom=0, top=1).subplots())
for ax in axs.flat:
    ax.set_xlim(-0.5, 4)
    ax.set_axis_off()
for ax in axs[0, :]:
    ax.text(-0.25, 0.5, "arrowstyle", size="large", color="tab:blue")
    ax.text(1.25, .5, "default parameters", size="large")
for ax, (stylename, stylecls) in zip(axs[1:, :].T.flat, styles.items()):
    # draw dot and annotation with arrowstyle
    l, = ax.plot(1, 0, "o", color="grey")
    ax.annotate(stylename, (1, 0), (0, 0),
                size="large", color="tab:blue", ha="center", va="center",
                arrowprops=dict(
                    arrowstyle=stylename, connectionstyle="arc3,rad=-0.05",
                    color="k", shrinkA=5, shrinkB=5, patchB=l,
                ),
                bbox=dict(boxstyle="square", fc="w", ec="grey"))
    # draw default parameters
    # wrap at every nth comma (n = 1 or 2, depending on text length)
    s = str(inspect.signature(stylecls))[1:-1]
    n = 2 if s.count(',') > 3 else 1
    ax.text(1.25, 0,
            re.sub(', ', lambda m, c=itertools.count(1): m.group()
                   if next(c) % n else '\n', s),
            verticalalignment="center")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.ArrowStyle`
#    - ``matplotlib.patches.ArrowStyle.get_styles``
#    - `matplotlib.axes.Axes.annotate`
