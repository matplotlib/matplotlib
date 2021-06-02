Colorbars are now an instance of Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.colorbar.Colorbar` class now inherits from `.axes.Axes`,
meaning that all of the standard methods of ``Axes`` can be used
directly on the colorbar object itself rather than having to access the
``ax`` attribute. For example, ::

    cbar.set_yticks()

rather than ::

    cbar.ax.set_yticks()

We are leaving the ``cbar.ax`` attribute in place as a pass-through for now,
which just maps back to the colorbar object.
