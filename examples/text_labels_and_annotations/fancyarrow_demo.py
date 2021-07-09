"""
================================
Annotation arrow style reference
================================

Overview of the arrow styles available in `~.Axes.annotate`.
"""

import inspect

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

styles = mpatches.ArrowStyle.get_styles()
ncol = 2
nrow = (len(styles) + 1) // ncol
axs = (plt.figure(figsize=(4 * ncol, 1 + nrow))
       .add_gridspec(1 + nrow, ncol,
                     wspace=.5, left=.1, right=.9, bottom=0, top=1).subplots())
for ax in axs.flat:
    ax.set_axis_off()
for ax in axs[0, :]:
    ax.text(0, .5, "arrowstyle",
            transform=ax.transAxes, size="large", color="tab:blue",
            horizontalalignment="center", verticalalignment="center")
    ax.text(.5, .5, "default parameters",
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")
for ax, (stylename, stylecls) in zip(axs[1:, :].T.flat, styles.items()):
    l, = ax.plot(.35, .5, "ok", transform=ax.transAxes)
    ax.annotate(stylename, (.35, .5), (0, .5),
                xycoords="axes fraction", textcoords="axes fraction",
                size="large", color="tab:blue",
                horizontalalignment="center", verticalalignment="center",
                arrowprops=dict(
                    arrowstyle=stylename, connectionstyle="arc3,rad=-0.05",
                    color="k", shrinkA=5, shrinkB=5, patchB=l,
                ),
                bbox=dict(boxstyle="square", fc="w"))
    ax.text(.5, .5,
            str(inspect.signature(stylecls))[1:-1].replace(", ", "\n"),
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")

plt.show()

#############################################################################
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
