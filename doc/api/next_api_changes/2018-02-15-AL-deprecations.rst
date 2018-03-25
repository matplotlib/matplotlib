Deprecations
````````````
The following modules are deprecated:

- :mod:`matplotlib.compat.subprocess`. This was a python 2 workaround, but all
  the functionality can now be found in the python 3 standard library
  :mod:`subprocess`.
- :mod:`matplotlib.backends.wx_compat`. Python 3 is only compatible with
  wxPython 4, so support for wxPython 3 or earlier can be dropped.

The following classes, methods, functions, and attributes are deprecated:

- ``Annotation.arrow``,
- ``cbook.GetRealpathAndStat``, ``cbook.Locked``,
- ``cbook.is_numlike`` (use ``isinstance(..., numbers.Number)`` instead),
  ``cbook.unicode_safe``
- ``container.Container.set_remove_method``,
- ``dates.DateFormatter.strftime_pre_1900``, ``dates.DateFormatter.strftime``,
- ``font_manager.TempCache``,
- ``mathtext.unichr_safe`` (use ``chr`` instead),
- ``FigureCanvasWx.macros``,
- ``texmanager.dvipng_hack_alpha``,

The following rcParams are deprecated:
- ``pgf.debug`` (the pgf backend relies on logging),

The the two-argument forms of ``cycler(label, values)`` and
``Axes.set_prop_cycle(label, values)`` are deprecated. Please use the keyword
syntax ``cycler(label=values)``, ``set_prop_cycle(label=values)`` instead.