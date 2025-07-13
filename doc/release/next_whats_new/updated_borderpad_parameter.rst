``borderpad`` accepts a tuple for separate x/y padding
-------------------------------------------------------

The ``borderpad`` parameter used for placing anchored artists (such as inset axes) now accepts a tuple of ``(x_pad, y_pad)``.

This allows for specifying separate padding values for the horizontal and
vertical directions, providing finer control over placement. For example, when
placing an inset in a corner, one might want horizontal padding to avoid
overlapping with the main plot's axis labels, but no vertical padding to keep
the inset flush with the plot area edge.

Example usage with :func:`~mpl_toolkits.axes_grid1.inset_locator.inset_axes`:

.. code-block:: python

    ax_inset = inset_axes(
        ax, width="30%", height="30%", loc='upper left',
        borderpad=(4, 0))
