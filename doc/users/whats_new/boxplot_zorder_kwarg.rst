Boxplot Zorder Keyword Argument
-------------------------------

The ``zorder`` parameter now exists for :func:`boxplot`. This allows the zorder
of a boxplot to be set in the plotting function call.

Example
```````
::

    boxplot(np.arange(10), zorder=10)
