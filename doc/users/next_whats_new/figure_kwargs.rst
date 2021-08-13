
Figure init passes keyword arguments through to set
---------------------------------------------------

Similar to many other sub-classes of `~.Artist`, `~.FigureBase`, `~.SubFigure`,
and `~.Figure` will now pass any additional keyword arguments to `~.Artist.set`
to allow properties of the newly created object to be set at init time.  For
example ::

  from matplotlib.figure import Figure
  fig = Figure(label='my figure')
