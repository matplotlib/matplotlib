
########################################
Arranging and laying out Axes (subplots)
########################################

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(3.5, 2.5),
                            layout="constrained")
    # add an artist, in this case a nice label in the middle...
    for row in range(2):
        for col in range(2):
            axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
                                transform=axs[row, col].transAxes,
                                ha='center', va='center', fontsize=18,
                                color='darkgrey')
    fig.suptitle('plt.subplots()')


.. toctree::
    :maxdepth: 1

    ../../../tutorials/intermediate/arranging_axes.rst
    ../../../gallery/subplots_axes_and_figures/colorbar_placement.rst
    ../../../gallery/subplots_axes_and_figures/mosaic.rst
    ../../../tutorials/intermediate/constrainedlayout_guide.rst
    ../../../tutorials/intermediate/tight_layout_guide.rst

