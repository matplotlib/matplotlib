New keyword argument 'sep' for EngFormatter
-------------------------------------------

A new "sep" keyword argument has been added to
:class:`~matplotlib.ticker.EngFormatter` and provides a means to define
the string that will be used between the value and its unit. The default
string is " ", which preserves the former behavior. Besides, the separator is
now present between the value and its unit even in the absence of SI prefix.
There was formerly a bug that was causing strings like "3.14V" to be returned
instead of the expected "3.14 V" (with the default behavior).
