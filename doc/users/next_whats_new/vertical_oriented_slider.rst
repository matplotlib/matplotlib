Adjusted ``matplotlib.widgets.Slider`` to have vertical orientation
-------------------------------------------------------------------

The :class:`matplotlib.widgets.Slider` widget now takes an optional argument
`orientation` which indicates the direction (`'horizontal'` or `'vertical'`)
that the slider should take.

Argument checking is in keeping with the existing code, and the actual changes
to the source are minimal, replacing `hspan`s, `hline`s and `xdata` with an if
switch to `vspan`, `vline`s and `ydata`.
