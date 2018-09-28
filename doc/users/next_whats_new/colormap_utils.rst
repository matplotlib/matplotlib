Colormap Utilities
------------------

Tools for joining, truncating, and resampling colormaps have been added. This grew out of https://gist.github.com/denis-bz/8052855, and http://stackoverflow.com/a/18926541/2121597.


Joining Colormaps
~~~~~~~~~~~~~~~~~

This includes the :func:`~matplotlib.colors.join_colormaps` function::

  import matplotlib.pyplat as plt
  from matplotlib.colors import join_colormaps

  viridis = plt.get_cmap('viridis', 128)
  plasma = plt.get_cmap('plasma_r', 64)
  jet = plt.get_cmap('jet', 64)
  
  joined_cmap = join_colormaps((viridis, plasma, jet))

This functionality has also been incorporated into the :meth:`~matplotlib.colors.colormap.join` and `~matplotlib.colors.colormap.__add__` methods, so that you can do things like::

  plasma_jet = plasma.join(jet)

  joined_cmap = viridis + plasma + jet  # Same as `join_colormaps` function call above

Truncating and resampling colormaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
A :meth:`~matplotlib.colors.colormap.truncate` method has also been added::

  sub_viridis = viridis.truncate(0.3, 0.8)

This gives a new colormap that goes from 30% to 80% of viridis. This functionality has also been implemented in the `~matplotlib.colors.colormap.__getitem__` method, so that the same colormap can be created by::

  sub_viridis = viridis[0.3:0.8]

The `~matplotlib.colors.colormap.__getitem__` method also supports a range of other 'advanced indexing' options, including integer slice indexing::

  sub_viridis2 = viridis[10:90:2]

integer list indexing, which may be particularly useful for creating discrete (low-N) colormaps::

  sub_viridis3 = viridis[[4, 35, 59, 90, 110]]

and `numpy.mgrid` style complex indexing::

  sub_viridis4 = viridis[0.2:0.4:64j]

See the `~matplotlib.colors.colormap.__getitem__` documentation for more details and examples of how to use these advanced indexing options.

Together, the join and truncate/resample methods allow the user to quickly construct new colormaps from existing ones::

  new_cm = viridis[0.5:] + plasma[:0.3] + jet[0.2:0.5:64j]

I doubt this colormap will ever be useful to someone, but hopefully it gives you the idea.
