``PolarAxes.get_rlim()`` and ``get_thetalim()`` added
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~matplotlib.projections.polar.PolarAxes` now provides
`~matplotlib.projections.polar.PolarAxes.get_rlim` and
`~matplotlib.projections.polar.PolarAxes.get_thetalim` to complement the
existing `~matplotlib.projections.polar.PolarAxes.set_rlim` and
`~matplotlib.projections.polar.PolarAxes.set_thetalim`. Previously, one
had to use `.Axes.get_ylim`, `.Axes.get_xlim` as a workaround.

::

    ax = plt.subplot(projection="polar")
    ax.set_rlim(1, 5)
    rmin, rmax = ax.get_rlim()   # was: AttributeError
