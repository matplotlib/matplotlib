*********************
``matplotlib.colors``
*********************

.. note::

   The Color :ref:`tutorials <tutorials-colors>` and :ref:`examples
   <color_examples>` demonstrate how to set colors and colormaps. You may want
   to read those instead.

.. currentmodule:: matplotlib.colors

.. automodule:: matplotlib.colors
   :no-members:
   :no-inherited-members:

Color Conversion tools
----------------------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   is_color_like
   same_color

   hsv_to_rgb
   rgb_to_hsv

   to_hex
   to_rgb
   to_rgba
   to_rgba_array

   get_named_colors_mapping


.. _norms_and_colormaps:

Normalization and Colormapping
------------------------------

Some Artists can map an array of input data to RGBA values, (ex
`~.axes.Axes.scatter` or `~.axes.Axes.imshow`).  The machinery for
this is implemented via the `~.cm.ScalarMappable` base class in `~.cm`
and the `~.Normalize` and `~.Colormap` classes in this module.

At the core, colormapping is going from a scalar value to a RGB(A) tuple
(formally :math:`f(x) : ℝ^1 \rightarrow ℝ^3`), that is we are tracing some
1D path through the 3D RGB space.  This mapping can be separated into two
orthogonal parts:

1. the path through color space
2. the mapping between the users data to distance along the curve.

The first step is expressed in Matplotlib via the `.Colormap` family
of classes and the second step is expressed through the `.Normalize` family
of classes.  This allows us to fully independently pick what colors to use (by
selecting the colormap), what data range to show (via the ``vmin`` and ``vmax``
attributes on `.Normalize` or the `.cm.ScalarMappable.set_clim` method), and
the functional transform (ex linear vs log) from data space to distance along the
curve space.

In addition to the colors in the map `.Colormap` objects carry three additional colors

- over (`.Colormap.set_over` / `.Colormap.get_over`)
- under (`.Colormap.set_under` / `.Colormap.get_under`)
- bad (`.Colormap.set_bad` / `.Colormap.get_bad`)

The first two (over / under) control what should be done for values
that are greater or less than the data range set by the user.  By
default these are equal to the top and bottom colors of the color map.
The "bad" value is used for masked or non-finite values (e.g. nan and
inf) and defaults to transparent.


.. note::

   Using `.cm.get_cmap` may return to you a reference to a globally
   visible instance of the colormap (rather than a new instance).  If
   you plan to set the over/under/bad values we recommend you first
   make a copy ::

     from copy import copy
     import matplotlib.cm as mcm

     my_cmap = copy(mcm.get_cmap('viridis'))


Both `.Colormap` and `.Normalize` are implemented as `callable classes
<https://docs.python.org/3/reference/datamodel.html#object.__call__>`__ which
allows use to bind some (mutable) state to a function call.  The `.Colormap.__call__`
signature when passed floats ::

  def map(X: NormedData, alpha:Optional[float] =None, bytes:Bool=False) -> RGBA:
     ...

Takes data in a "normalized" space and:

- maps values in the closed set ``[0, 1]`` to that fraction along the curve
- maps any value greater than 1 to the "over" color
- maps any value less than 0 to the "under" color
- maps any non-finite or masked value to the "bad" color

broadcasting to match the input shape (scalar to tuple, n-D array to
(n+1)-D array).  This can be useful to get a set of colors drawn from
a color map ::

  import matplotlib.cm as mcm
  import numpy as np

  cmap = mcm.get_cmap('viridis')
  list_of_colors = cmap(np.arange(0, 1, 5))



.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Colormap
   LinearSegmentedColormap
   ListedColormap


.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Normalize
   LogNorm
   CenteredNorm
   BoundaryNorm
   TwoSlopeNorm
   PowerNorm
   NoNorm
   SymLogNorm

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   from_levels_and_colors
   make_norm_from_scale


.. inheritance-diagram:: matplotlib.colors.Colormap matplotlib.colors.LinearSegmentedColormap matplotlib.colors.ListedColormap
   :parts: 1
   :private-bases:


.. inheritance-diagram:: matplotlib.colors.Normalize matplotlib.colors.LogNorm matplotlib.colors.PowerNorm matplotlib.colors.NoNorm matplotlib.colors.TwoSlopeNorm matplotlib.colors.SymLogNorm matplotlib.colors.BoundaryNorm
   :parts: 1
   :private-bases:


Hill Shading
------------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   LightSource
