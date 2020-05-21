"""
================
Nested Subpanels
================

Sometimes it is desirable to have a figure that has two different
layouts in it.  This can be achieved with
:doc:`nested gridspecs</examples/subplots_axes_and_figures/gridspec_nested>`
but having a virtual figure with its own artists is helpful, so
Matplotlib also has "subpanels", usually implimented by calling
``.figure.PanelBase.add_subpanel`` in a way that is analagous to
``.figure.PanelBase.add_subplot``.

"""
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12, hide_labels=False):
    ax.pcolormesh(np.random.randn(30, 30))
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)

# gridspec inside gridspec
fig = plt.figure(constrained_layout=True)
subpanels = fig.subpanels(1, 2, wspace=0.07)

axsLeft = subpanels[0].subplots(1, 2, sharey=True)
subpanels[0].set_facecolor('0.75')
for ax in axsLeft:
    example_plot(ax)
subpanels[0].suptitle('Left plots', fontsize='x-large')

axsRight = subpanels[1].subplots(3, 1, sharex=True)
for nn, ax in enumerate(axsRight):
    example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')
subpanels[1].suptitle('Right plots', fontsize='x-large')

fig.suptitle('Figure suptitle', fontsize='xx-large')

plt.show()

