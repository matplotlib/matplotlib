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

Normalization and Colormapping of Continuous Data
-------------------------------------------------

Some `~.artist.Artist` classes can map an array of input data to RGBA
values, (ex `~.axes.Axes.scatter` or `~.axes.Axes.imshow`).  The
machinery for this is implemented via the `~.cm.ScalarMappable` base
class in `~.cm` and the `~.Normalize` and `~.Colormap` classes in
`~.colors` (this module).

At the core, colormapping is going from a scalar value to a RGB tuple (formally
:math:`f(x) : ℝ^1 \rightarrow ℝ^3`).  To effectively communicate through the
color we want pick a :ref:`colormap suited to the data
<sphx_glr_tutorials_colors_colormaps.py>`.  For continuous data types [#f1]_ a
"good" colormap smoothly and continuously change their RGB values as a function
of the input that trace out a 1-dimensional path through the 3-dimensional RGB
space [#f2]_.  We can restrict the domain of :math:`f` to $[0, 1]$ which we
interpret as the normalized distance along the curve.  This allows us to
cleanly separate the mapping process from the continuous input data to RGB into
two steps:

1. the mapping between the user's data to distance along the curve
2. the parameterized path through color space.

The first step is expressed through the `.Normalize` family of classes and the
second is expressed in Matplotlib via the `.Colormap` family of classes.  This
allows us to fully independently pick the functional transform (e.g., linear vs
log) from data space to distance along the curve space, what (user) data range
to show (via the ``vmin`` and ``vmax`` attributes on `.Normalize`, or via the
`.cm.ScalarMappable.set_clim` method), and what colors to use (by selecting the
`.Colormap`).  Both `.Colormap` and `.Normalize` are implemented as `callable
classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`__
which allows use to bind some (mutable) state to a function call. The complete
functionality is exposed in the `.ScalarMappable` family of artists which have
a `.Colormap` and `.Normalize` instances and are responsible for invoking them
at draw time.

The `.Normalize` family has 3 common attributes: *vmin*, *vmax*, and *clip*
which control the data limits.  The `.Normalize.__call__` signature is ::

  def __call__(value: RawData, clip:Optional[Bool] =None) -> NormedData:
     ...

It takes in data in the user's data space and converts it to *NormedData* with
the range:

.. math::

     \begin{cases}
      \mathrm{under} & d <  vmin \\
      [0, 1] & vmin \leq d \leq vmax \\
      \mathrm{over} & vmax < d \\
      \mathrm{bad} & !\mathrm{np.finite(d)}
    \end{cases}



The `.Colormap.__call__` signature when passed *NormedData* (floats) [#f3]_ is
::

  def __call__(self, X: NormedData,
               alpha:Optional[float] =None, bytes:Bool=False) -> RGBA:
     ...

In addition to parameterized path through RGB (which handles values in $[0,
1]$,  `.Colormap` objects carry three additional colors:

- *over* (`.Colormap.set_over` / `.Colormap.get_over`)
- *under* (`.Colormap.set_under` / `.Colormap.get_under`)
- *bad* (`.Colormap.set_bad` / `.Colormap.get_bad`)

which control the color for the corresponding values in *NormedData*.
By default the over and under colors are the top and bottom colors of
the colormap respectively and bad is transparent.

.. warning::

   Using `.cm.get_cmap` may return to you a reference to a globally
   visible instance of the colormap (rather than a new instance).  If
   you plan to set the over/under/bad values we recommend you first
   make a copy ::

     from copy import copy
     import matplotlib.cm as mcm

     my_cmap = copy(mcm.get_cmap('viridis'))

.. rubric:: Footnotes

.. [#f1] Discrete data types, such as Categorical and Ordinal, have different
         considerations.
.. [#f2] Notable, the cubehelix colormap is named because it traces a helix
         through the RGB color cube from black to white.
.. [#f3] Discrete data, as handled by `.NoNorm` and `.BoundaryNorm` are passed
         as integers and act as direct Look Up Table (LUT) indexes into the
         colormap.

In practice
~~~~~~~~~~~

To make the above section concrete, lets first consider the linear `.Normalize`

.. ipython ::

   In [104]: import matplotlib.colors as mcolors

   In [105]: norm = mcolors.Normalize(vmin=100, vmax=300)

   In [106]: norm
   Out[106]: <matplotlib.colors.Normalize at 0x7f9bf441aeb0>

If we now pass in values in the range of :math:`[vmin, vmax]`

.. ipython ::


   In [130]: norm([100, 200, 300])
   Out[130]:
   masked_array(data=[ 0. , 0.5, 1.],
                mask=False,
          fill_value=1e+20)

We see that they are scaled as expected.  If we also pass in some over
/ under / bad values

.. ipython ::

   In [131]: norm([0, 100, 200, 300, 400, np.nan])
   Out[131]:
   masked_array(data=[-0.5,  0. ,  0.5,  1. ,  1.5,  nan],
                mask=False,
          fill_value=1e+20)

we see that they are also scaled and produce values outside of the
range :math:`[0, 1]`.  If you need the values to be clipped, you can do
have the norm do that for you via the *clip* kwarg

.. ipython ::

   In [130]: norm([0, 100, 200, 300, 400, np.nan], clip=True)
   Out[130]:
   masked_array(data=[0. , 0. , 0.5, 1. , 1. , nan],
                mask=False,
          fill_value=1e+20)

The default value of *clip* can be set when instantiating the
`.Normalize` instance.

We can also use a non-linear norm

.. ipython ::

   In [136]: log_norm = mcolors.LogNorm(10, 1000)

   In [137]: log_norm([10, 100, 1000])
   Out[137]:
   masked_array(data=[0.0, 0.5, 1.0],
                mask=[False, False, False],
          fill_value=1e+20)

if the data has a large dynamic range.

Once we have normalized our data we can pass it to the colormap

.. ipython ::

   In [102]: import copy

   In [103]: import matplotlib.cm as mcm

   In [141]: viridis = copy.copy(mcm.get_cmap('viridis'))

.. ipython ::

   In [143]: viridis([0, .5, 1])
   Out[143]:
   array([[0.267004, 0.004874, 0.329415, 1.      ],
          [0.127568, 0.566949, 0.550556, 1.      ],
          [0.993248, 0.906157, 0.143936, 1.      ]])

Which pulls out the bottom, middle, and top color of the *viridis* .
If we set the over/under/bad colors and pass out-of-range values we can also
see them pulled out:

.. ipython ::

   In [144]: viridis.set_over('w')

   In [147]: viridis.set_under('k')

   In [148]: viridis.set_bad('r')

   In [149]: viridis([1.5, -0.5, np.nan])
   Out[149]:
   array([[1., 1., 1., 1.],
          [0., 0., 0., 1.],
          [1., 0., 0., 1.]])



Directly using a `.Colormap` outside of a `.ScalarMappable` can be useful
to generate a family of coherent colors for plotting

.. plot::
    :include-source:

    import matplotlib.cm as mcm
    import numpy as np

    cmap = mcm.get_cmap('viridis')
    array_of_colors = cmap(np.linspace(0, 1, 5))

    x = np.linspace(0, 1, 25)
    fig, ax = plt.subplots(constrained_layout=True)
    for j, color in enumerate(array_of_colors):
        ax.plot(x, x**j, color=color, label=f'$x^{j}$')
    ax.legend()

API
~~~

Colormap Classes
++++++++++++++++

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Colormap
   LinearSegmentedColormap
   ListedColormap


.. inheritance-diagram:: matplotlib.colors.Colormap matplotlib.colors.LinearSegmentedColormap matplotlib.colors.ListedColormap
   :parts: 1
   :private-bases:


Norm Classes
++++++++++++

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Normalize
   LogNorm
   CenteredNorm
   TwoSlopeNorm
   PowerNorm
   SymLogNorm
   FuncNorm
   BoundaryNorm
   NoNorm


.. inheritance-diagram:: matplotlib.colors.Normalize matplotlib.colors.LogNorm matplotlib.colors.PowerNorm matplotlib.colors.NoNorm matplotlib.colors.TwoSlopeNorm matplotlib.colors.SymLogNorm matplotlib.colors.BoundaryNorm matplotlib.colors.FuncNorm matplotlib.colors.CenteredNorm
   :parts: 1
   :private-bases:


Factory Functions & Decorators
++++++++++++++++++++++++++++++


.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   from_levels_and_colors
   make_norm_from_scale



Hill Shading
------------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   LightSource
