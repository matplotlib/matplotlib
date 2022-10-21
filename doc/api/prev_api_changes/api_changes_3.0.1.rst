API Changes for 3.0.1
=====================

``matplotlib.tight_layout.auto_adjust_subplotpars`` can return ``None`` now if
the new subplotparams will collapse axes to zero width or height.
This prevents ``tight_layout`` from being executed.  Similarly
``matplotlib.tight_layout.get_tight_layout_figure`` will return None.

To improve import (startup) time, private modules are now imported lazily.
These modules are no longer available at these locations:

- ``matplotlib.backends.backend_agg._png``
- ``matplotlib.contour._contour``
- ``matplotlib.image._png``
- ``matplotlib.mathtext._png``
- ``matplotlib.testing.compare._png``
- ``matplotlib.texmanager._png``
- ``matplotlib.tri.triangulation._tri``
- ``matplotlib.tri.triangulation._qhull``
- ``matplotlib.tri.tricontour._tri``
- ``matplotlib.tri.trifinder._tri``
