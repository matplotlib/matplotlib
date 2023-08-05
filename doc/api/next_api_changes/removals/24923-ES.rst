cbook removals
~~~~~~~~~~~~~~

- ``matplotlib.cbook.MatplotlibDeprecationWarning`` and
  ``matplotlib.cbook.mplDeprecation`` are removed; use
  `matplotlib.MatplotlibDeprecationWarning` instead.
- ``cbook.maxdict``; use the standard library ``functools.lru_cache`` instead.

Groupers from ``get_shared_x_axes`` / ``get_shared_y_axes`` are immutable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modifications to the Groupers returned by ``get_shared_x_axes`` and
``get_shared_y_axes`` are no longer allowed. Note that previously, calling e.g.
``join()`` would already fail to set up the correct structures for sharing
axes; use `.Axes.sharex` or `.Axes.sharey` instead.
