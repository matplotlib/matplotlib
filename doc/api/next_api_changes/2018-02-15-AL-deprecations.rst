Deprecations
````````````
The following modules are deprecated:

- :mod:`matplotlib.compat.subprocess`. This was a python 2 workaround, but all the
  functionality can now be found in the python 3 standard library
  :mod:`subprocess`.

The following functions and classes are deprecated:

- ``cbook.GetRealpathAndStat`` (which is only a helper for
  ``get_realpath_and_stat``),
- ``cbook.Locked``,
- ``cbook.is_numlike`` (use ``isinstance(..., numbers.Number)`` instead),
- ``mathtext.unichr_safe`` (use ``chr`` instead),
- ``texmanager.dvipng_hack_alpha``,

The following rcParams are deprecated:
- ``pgf.debug`` (the pgf backend relies on logging),
