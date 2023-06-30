``hexbin`` now defaults to ``rcParams["patch.linewidth"]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default value of the *linewidths* argument of `.Axes.hexbin` has
been changed from ``1.0`` to :rc:`patch.linewidth`. This improves the
consistency with `.QuadMesh` in `.Axes.pcolormesh` and `.Axes.hist2d`.
