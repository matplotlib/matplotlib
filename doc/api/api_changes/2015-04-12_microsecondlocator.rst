Removed `args` and `kwargs` from `MicrosecondLocator.__call__`
``````````````````````````````````````````````````````````````

The call signature of :meth:`~matplotlib.dates.MicrosecondLocator.__call__`
has changed from `__call__(self, *args, **kwargs)` to `__call__(self)`.
This is consistent with the super class :class:`~matplotlib.ticker.Locator`
and also all the other Locators derived from this super class.


No `ValueError` for the MicrosecondLocator and YearLocator
``````````````````````````````````````````````````````````

The :class:`~matplotlib.dates.MicrosecondLocator` and
:class:`~matplotlib.dates.YearLocator` objects when called will return
an empty list if the axes have no data or the view has no interval.
Previously, they raised a `ValueError`. This is consistent with all
the Date Locators.
