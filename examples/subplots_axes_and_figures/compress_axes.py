"""
===========================
Compress axes layout option
===========================

If a grid of subplot axes have fixed aspect ratios, the axes will usually
be too far apart in one of the dimensions.  For simple layouts
the ``compress_layout=True`` option to `.Figure.figure` or `.subplots` can
try to compress that dimension so the axes are a similar distance apart in
both dimensions.
"""

import matplotlib.pyplot as plt
import numpy as np

#############################################################################
#
# The default behavior with constrained_layout.  Note how there is a large
# horizontal space between the axes.

fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                        sharex=True, sharey=True, constrained_layout=True)
for ax in axs.flat:
    ax.set_aspect(1.0)
plt.show()

#############################################################################
#
# Adding ``compress_layout=True`` attempts to collapse this space:

fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                        sharex=True, sharey=True, constrained_layout=True,
                        compress_layout=True)
for ax in axs.flat:
    ax.set_aspect(1.0)
plt.show()

#############################################################################
#
# Compatibility
# -------------
#
# Currently this works with ``constrained_layout=True`` for simple layouts
# that do not have nested gridspec layouts
# (:doc:`gallery/subplots_axes_and_figures/gridspec_nested.html`).  This
# includes simple colorbar layouts with ``constrained_layout``:

fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                        sharex=True, sharey=True, constrained_layout=True,
                        compress_layout=True)
for ax in axs.flat:
    ax.set_aspect(1.0)
    pc = ax.pcolormesh(np.random.randn(20,20))
    fig.colorbar(pc, ax=ax)
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                        sharex=True, sharey=True, constrained_layout=True,
                        compress_layout=True)
for ax in axs.flat:
    ax.set_aspect(1.0)
    pc = ax.pcolormesh(np.random.randn(20,20))
fig.colorbar(pc, ax=axs)
plt.show()

#############################################################################
# Compatibility is currently not as good with ``tight_layout`` or no layout
# manager, primarily because colorbars are implimented with nested gridspecs.

for tl in [True, False]:
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                            sharex=True, sharey=True, tight_layout=tl,
                            compress_layout=True)
    for ax in axs.flat:
        ax.set_aspect(1.0)
        pc = ax.pcolormesh(np.random.randn(20,20))
        fig.colorbar(pc, ax=ax)
        fig.suptitle(f'Tight Layout: {tl}')
plt.show()

#############################################################################
# However, both work with simple layouts that do not have colorbars. 

for tl in [True, False]:
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), facecolor='0.75',
                            sharex=True, sharey=True, tight_layout=tl,
                            compress_layout=True)
    for ax in axs.flat:
        ax.set_aspect(1.0)
        pc = ax.pcolormesh(np.random.randn(20,20))
        # fig.colorbar(pc, ax=ax)
        fig.suptitle(f'Tight Layout: {tl}')
plt.show()
