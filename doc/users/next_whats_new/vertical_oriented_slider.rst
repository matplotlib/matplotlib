Adjusted ``matplotlib.widgets.Slider`` to have vertical orientation
-------------------------------------------------------------------

The :class:`matplotlib.widgets.Slider` widget now takes an optional argument
``orientation`` which indicates the direction (``'horizontal'`` or ``'vertical'``)
that the slider should take.

Argument checking is in keeping with the existing code, and the actual changes
to the source are minimal, replacing ``hspan``, ``hline`` and ``xdata`` with an if
switch to ``vspan``, ``vline`` and ``ydata``.

Inspired by https://stackoverflow.com/questions/25934279/add-a-vertical-slider-with-matplotlib
