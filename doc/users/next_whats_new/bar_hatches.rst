A list of hatches can be specified to `~.axes.Axes.bar` and `~.axes.Axes.barh`
------------------------------------------------------------------------------

Similar to some other rectangle properties, it is now possible to hand a list
of hatch styles to `~.axes.Axes.bar` and `~.axes.Axes.barh` in order to create
bars with different hatch styles, e.g.

.. plot::

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()
  ax.bar([1, 2], [2, 3], hatch=['+', 'o'])
  plt.show()
