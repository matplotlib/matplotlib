Callable *valfmt* for ``Slider`` and ``RangeSlider``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the existing %-format string, the *valfmt* parameter of
`~.matplotlib.widgets.Slider` and `~.matplotlib.widgets.RangeSlider` now
also accepts a callable of the form ``valfmt(val: float) -> str``.
