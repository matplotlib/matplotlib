DateFormatter strftime
----------------------
Date formatters' (:class:`~matplotlib.dates.DateFormatter`)
:meth:`~matplotlib.dates.DateFormatter.strftime` method will format
a :class:`datetime.datetime` object with the format string passed to
the formatter's constructor. This method accepts datetimes with years
before 1900, unlike :meth:`datetime.datetime.strftime`.
