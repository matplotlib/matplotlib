"""
================
Nested Gridspecs
================

GridSpecs can be nested, so that a subplot from a parent GridSpec can
set the position for a nested grid of subplots.

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
f = plt.figure()

gs0 = gridspec.GridSpec(1, 2, figure=f)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

ax1 = f.add_subplot(gs00[:-1, :])
ax2 = f.add_subplot(gs00[-1, :-1])
ax3 = f.add_subplot(gs00[-1, -1])

# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs01 = gs0[1].subgridspec(3, 3)

ax4 = f.add_subplot(gs01[:, :-1])
ax5 = f.add_subplot(gs01[:-1, -1])
ax6 = f.add_subplot(gs01[-1, -1])

plt.suptitle("GridSpec Inside GridSpec")
format_axes(f)

plt.show()
