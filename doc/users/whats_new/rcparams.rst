Configuration (rcParams)
------------------------

+---------------------------------+--------------------------------------------------+
| Parameter                       | Description                                      |
+=================================+==================================================+
|`date.autoformatter.year`        | foramt string for 'year' scale dates             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.month`       | format string for 'month' scale dates            |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.day`         | format string for 'day' scale dates              |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.hour`        | format string for 'hour' scale times             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.minute`      | format string for 'minute' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.second`      | format string for 'second' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.microsecond` | format string for 'microsecond' scale times      |
+---------------------------------+--------------------------------------------------+
|`scatter.marker`                 | default marker for scatter plot                  |
+---------------------------------+--------------------------------------------------+
|`svg.hashsalt`                   | see note                                         |
+---------------------------------+--------------------------------------------------+

Added ``svg.hashsalt`` key to rcParams
```````````````````````````````````````

If ``svg.hashsalt`` is ``None`` (which it is by default), the svg
backend uses ``uuid4`` to generate the hash salt.  If it is not
``None``, it must be a string that is used as the hash salt instead of
``uuid4``.  This allows for deterministic SVG output.


Removed the ``svg.image_noscale`` rcParam
`````````````````````````````````````````

As a result of the extensive changes to image handling, the
``svg.image_noscale`` rcParam has been removed.  The same
functionality may be achieved by setting ``interpolation='none'`` on
individual images or globally using the ``image.interpolation``
rcParam.
