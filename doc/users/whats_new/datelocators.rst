Date Locators
-------------

Date Locators (derived from :class:`~matplotlib.dates.DateLocator`) now
implement the :meth:`~matplotlib.tickers.Locator.tick_values` method.
This is expected of all Locators derived from :class:`~matplotlib.tickers.Locator`.

The Date Locators can now be used easily without creating axes

    from datetime import datetime
    from matplotlib.dates import YearLocator
    t0 = datetime(2002, 10, 9, 12, 10)
    tf = datetime(2005, 10, 9, 12, 15)
    loc = YearLocator()
    values = loc.tick_values(t0, tf)
