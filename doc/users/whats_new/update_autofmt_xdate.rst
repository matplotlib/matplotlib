New which Parameter for autofmt_xdate
-------------------------------------

A ``which`` parameter now exists for the method :func:`autofmt_xdate`. This
allows a user to format ``major``, ``minor`` or ``both`` tick labels
selectively. If ``which`` is ``None`` (default) then the method will rotate
``major`` tick labels.

Example
```````
::

    autofmt_xdate(self, bottom=0.2, rotation=30, ha='right', which='minor')
