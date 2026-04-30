``PolarAxes.get_rlim()`` and ``get_thetalim()`` added
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~matplotlib.projections.polar.PolarAxes` now provides
`~matplotlib.projections.polar.PolarAxes.get_rlim` and
`~matplotlib.projections.polar.PolarAxes.get_thetalim` to complement the
existing `~matplotlib.projections.polar.PolarAxes.set_rlim` and
`~matplotlib.projections.polar.PolarAxes.set_thetalim`. Previously, calling
these getters raised an ``AttributeError``; the workaround was to call the
base-class ``get_ylim()`` / ``get_xlim()`` directly::

    ax = plt.subplot(projection="polar")
    ax.set_rlim(1, 5)
    rmin, rmax = ax.get_rlim()   # was: AttributeError
