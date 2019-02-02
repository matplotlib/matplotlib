API Changes
```````````

The first parameter of `matplotlib.use` has been renamed from *arg* to
*backend*. This will only affect cases where that parameter has been set
as a keyword argument. The common usage pattern as a positional argument
``matplotlib.use('Qt5Agg')`` is not affected.
