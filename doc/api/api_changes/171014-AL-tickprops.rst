Deprecation of redundant `Tick` attributes
``````````````````````````````````````````

The ``gridOn``, ``tick1On``, ``tick2On``, ``label1On``, and ``label2On``
`~.Tick` attributes have been deprecated.  Directly get and set the visibility
on the underlying artists, available as the ``gridline``, ``tick1line``,
``tick2line``, ``label1``, and ``label2`` attributes.

The ``label`` attribute, which was an alias for ``label1``, has been
deprecated.

Subclasses that relied on setting the above visibility attributes needs to be
updated; see e.g. :file:`examples/api/skewt.py`.
