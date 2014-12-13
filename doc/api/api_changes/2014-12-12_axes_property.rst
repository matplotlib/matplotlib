Prevent moving artists between Axes, Property-ify Artist.axes, deprecate Artist.{get,set}_axes
``````````````````````````````````````````````````````````````````````````````````````````````

The reason this was done was to prevent adding an Artist that is
already associated with an Axes to be moved/added to a different Axes.
This was never supported as it causes havoc with the transform stack.
The apparent support for this (as it did not raise an exception) was
the source of multiple bug reports and questions on SO.

For almost all use-cases, the assignment of the axes to an artist should be
taken care of by the axes as part of the ``Axes.add_*`` method, hence the
deprecation {get,set}_axes.

Removing the ``set_axes`` method will also remove the 'axes' line from
the ACCEPTS kwarg tables (assuming that the removal date gets here
before that gets overhauled).
