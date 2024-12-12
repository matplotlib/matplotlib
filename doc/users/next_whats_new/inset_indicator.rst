``InsetIndicator`` artist
~~~~~~~~~~~~~~~~~~~~~~~~~

`~.Axes.indicate_inset` and `~.Axes.indicate_inset_zoom` now return an instance
of `~matplotlib.inset.InsetIndicator` which contains the rectangle and
connector patches.  These patches now update automatically so that

.. code-block:: python

    ax.indicate_inset_zoom(ax_inset)
    ax_inset.set_xlim(new_lim)

now gives the same result as

.. code-block:: python

    ax_inset.set_xlim(new_lim)
    ax.indicate_inset_zoom(ax_inset)
