Changes to behavior of Axes creation methods (``gca()``, ``add_axes()``, ``add_subplot()``)
-------------------------------------------------------------------------------------------

The behavior of the functions to create new axes (`.pyplot.axes`,
`.pyplot.subplot`, `.figure.Figure.add_axes`,
`.figure.Figure.add_subplot`) has changed. In the past, these functions would
detect if you were attempting to create Axes with the same keyword arguments as
already-existing axes in the current figure, and if so, they would return the
existing Axes. Now, these functions will always create new Axes. A special
exception is `.pyplot.subplot`, which will reuse any existing subplot with a
matching subplot spec. However, if there is a subplot with a matching subplot
spec, then that subplot will be returned, even if the keyword arguments with
which it was created differ.

Correspondingly, the behavior of the functions to get the current Axes
(`.pyplot.gca`, `.figure.Figure.gca`) has changed. In the past, these functions
accepted keyword arguments. If the keyword arguments matched an
already-existing Axes, then that Axes would be returned, otherwise new Axes
would be created with those keyword arguments. Now, the keyword arguments are
only considered if there are no axes at all in the current figure. In a future
release, these functions will not accept keyword arguments at all.
