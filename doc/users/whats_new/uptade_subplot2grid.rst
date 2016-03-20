New Firgure Parameter for subplot2grid
--------------------------------------

A ``fig`` parameter now exists for the method :func:`subplot2grid`.  This allows
for the figure that the subplots will be created in to be specified.  If ``fig``
is ``None`` (default) then the method will use the current figure retrieved by
:func:`gcf`.

Example
```````
::

    subplot2grid(shape, loc, rowspan=1, colspan=1, fig=myfig)
