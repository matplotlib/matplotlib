``Figure.add_axes()`` without parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.Figure.add_axes` can now be called without parameters to add a full-figure
`~.axes.Axes`. This is equivalent to calling ``fig.add_subplot()`` without
parameters. This also creates the same Axes as ``fig, ax = plt.subplots()``
does.
