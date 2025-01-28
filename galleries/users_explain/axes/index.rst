+++++++++++++++++
Axes and subplots
+++++++++++++++++

Matplotlib `~.axes.Axes` are the gateway to creating your data visualizations.
Once an Axes is placed on a figure there are many methods that can be used to
add data to the Axes. An Axes typically has a pair of `~.axis.Axis`
Artists that define the data coordinate system, and include methods to add
annotations like x- and y-labels, titles, and legends.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(3.5, 2.5),
                            layout="constrained")
    # for each Axes, add an artist, in this case a nice label in the middle...
    for row in range(2):
        for col in range(2):
            axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
                                transform=axs[row, col].transAxes,
                                ha='center', va='center', fontsize=18,
                                color='darkgrey')
    fig.suptitle('plt.subplots()')

.. toctree::
    :maxdepth: 2

    axes_intro

.. toctree::
    :maxdepth: 1

    arranging_axes
    colorbar_placement
    Autoscaling axes <autoscale>

.. toctree::
    :maxdepth: 2
    :includehidden:

    axes_scales
    axes_ticks
    axes_units
    Legends <legend_guide>
    Subplot mosaic <mosaic>

.. toctree::
    :maxdepth: 1
    :includehidden:

    Constrained layout guide <constrainedlayout_guide>
    Tight layout guide (mildly discouraged) <tight_layout_guide>
