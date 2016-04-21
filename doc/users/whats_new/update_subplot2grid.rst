New Figure Parameter for subplot2grid
--------------------------------------

A ``fig`` parameter now exists for the method :func:`subplot2grid`.  This allows
a user to specify the figure where the subplots will be created.  If ``fig``
is ``None`` (default) then the method will use the current figure retrieved by
:func:`gcf`.

Example
```````
::

    subplot2grid(shape, loc, rowspan=1, colspan=1, fig=myfig)
