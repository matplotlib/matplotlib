==============================
 Changes to the default style
==============================

The most important changes in matplotlib 2.0 are the changes to the
default style.

While it is impossible to select the best default for all cases, these
are designed to work well in the most common cases.

A 'classic' style sheet is provided so reverting to the 1.x default
values is a single line of python

.. code::

  mpl.style.use('classic')

See :ref:`customizing-with-matplotlibrc-files` for details about how to
persistently and selectively revert many of these changes.


colors, color cycles, and color maps
====================================

Colors in default property cycle
--------------------------------

The colors in the default proprety cycle have been changed from
``['b', 'g', 'r', 'c', 'm', 'y', 'k']`` to the `Vega
category10 palette
<https://github.com/vega/vega/wiki/Scales#scale-range-literals>`__

.. plot::


  th = np.linspace(0, 2*np.pi, 512)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))


  def color_demo(ax, colors, title):
      ax.set_title(title)
      for j, c in enumerate(colors):
          v_offset = -(j / len(colors))
          ax.plot(th, .1*np.sin(th) + v_offset, color=c)
          ax.annotate("'C{}'".format(j), (0, v_offset),
                      xytext=(-1.5, 0),
                      ha='right',
                      va='center',
                      color=c,
                      textcoords='offset points',
                      family='monospace')

          ax.annotate("{!r}".format(c), (2*np.pi, v_offset),
                      xytext=(1.5, 0),
                      ha='left',
                      va='center',
                      color=c,
                      textcoords='offset points',
                      family='monospace')
      ax.axis('off')

  old_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

  new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf']

  color_demo(ax1, old_colors, 'classic')
  color_demo(ax2, new_colors, 'v2.0')

  fig.subplots_adjust(**{'bottom': 0.0, 'left': 0.059, 'right': 0.869, 'top': 0.895})

In addition to changing the colors, an additional method to specify
colors was added.  Previously, the default colors were the single
character short-hand notations for red, green, blue, cyan, magenta,
yellow, and black.  This made them easy to type and usable in the
abbreviated style string in ``plot``.  On the other hand, the new
colors are only specified via a hex value.  To make in easy to access
these colors the notation for colors ``'CN'`` was added to access
colors.  This allows the first 10 colors in
``mpl.rcParms['axes.prop_cycle']`` to be easily accessed.  See
:ref:`colors` for more details.

To restore only the old color cycle use

.. code::

   from cycler import cycler
   mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

or setting

.. code::

   axes.prop_cycle    : cycler('color', 'bgrcmyk')

in your :file:`matplotlibrc` file.

Colormap
--------


The new default color map used by `matplotlib.cm.ScalarMappable` instances is
 `'viridis'` (aka `option D <http://bids.github.io/colormap/>`__).

.. plot::

   import numpy as np
   N = M = 200
   X, Y = np.ogrid[0:20:N*1j, 0:20:M*1j]
   data = np.sin(np.pi * X*2 / 20) * np.cos(np.pi * Y*2 / 20)

   fig, ax = plt.subplots()
   im = ax.imshow(data, extent=[0, 200, 0, 200])
   fig.colorbar(im)
   ax.set_title('viridis')

For an introduction to color theory and how 'viridis' was generated
watch Nathaniel Smith and Stéfan van der Walt's talk from SciPy2015.
See `here for many more deatils <http://bids.github.io/colormap/>`__
about the other alternatives and the tools used to create the color
map.  For details on all of color maps available in matplotlib see
:ref:`colormaps`.

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/xAoljeRJ3lU" frameborder="0" allowfullscreen></iframe>


The previous default can be restored using

.. code::

   mpl.rcParams['image.cmap'] = 'jet'

or setting

.. code::

   image.cmap    : 'jet'

in your :file:`matplotlibrc` file, however this is strongly discouraged.

Other colors
------------

- The default interactive figure background color has changed from
  grey to white.  Use the rcParam ``figure.facecolor`` to control
  this.

Grid lines
----------


- Grid lines are light grey solid 1pt lines.  They are no longer dashed by
  default.

Plots
=====
- For markers, scatter plots, bar charts and pie charts, there is no
  longer a black outline around filled markers by default.

- The default size of the elements in a scatter plot is now based on
  the rcParam ``lines.markersize`` so it is consistent with ``plot(X,
  Y, 'o')``.  The old value was 20, and the new value is 36 (6^2).

Hatching
========

- The width of the lines in a hatch pattern is now configurable by the
  rcParam `hatch.linewidth`, with a default of 1 point.  The old
  behavior was different depending on backend:

    - PDF: 0.1 pt
    - SVG: 1.0 pt
    - PS:  1 px
    - Agg: 1 px

Plot layout
===========

- The default dpi used for on-screen is now 100, which is the same as
  the old default for saving files.  Due to this, the on-screen
  display is now more what-you-see-is-what-you-get.

- The number of ticks on an axis is now automatically determined based
  on the length of the axis.

- The limits are scaled to exactly the dimensions of the data, plus 5%
  padding.  The old behavior was to scale to the nearest "round"
  numbers.  To use the old behavior, set the ``rcParam``
  ``axes.autolimit_mode`` to ``round_numbers``.  To control the
  margins on particular side individually, pass any of the following
  to any artist or plotting function:

  - ``top_margin=False``
  - ``bottom_margin=False``
  - ``left_margin=False``
  - ``right_margin=False``

- Ticks now point outward by default.  To have ticks pointing inward,
  use the ``rcParams`` ``xtick.direction`` and ``ytick.direction``.

- Ticks and grids are now plotted above solid elements such as
  filled contours, but below lines.  To return to the previous
  behavior of plotting ticks and grids above lines, set
  ``rcParams['axes.axisbelow'] = False``.

- By default, caps on the ends of errorbars are not present.  Use the
  rcParam ``errorbar.capsize`` to control this.

Images
======

- The default mode for image interpolation, in the rcParam
  ``image.interpolation``, is now ``nearest``.

- The default shading mode for light source shading, in
  ``matplotlib.colors.LightSource.shade``, is now ``overlay``.
  Formerly, it was ``hsv``.

- The default value for the rcParam ``image.resample`` is now
  ``True``.  This will apply interpolation for both upsampling and
  downsampling of an image.

Fonts
=====

- The default font has changed from "Bitstream Vera Sans" to "DejaVu
  Sans".  "DejaVu Sans" is an improvement on "Bistream Vera Sans" that
  adds more international and math characters, but otherwise has the
  same appearance.

- The default math font when using the built-in math rendering engine
  (mathtext) has changed from "Computer Modern" (i.e. LaTeX-like) to
  "DejaVu Sans".  To revert to the old behavior, set the ``rcParam``
  ``mathtext.fontset`` to ``cm``.  This change has no effect if the
  TeX backend is used (i.e. ``text.usetex`` is ``True``).

Dates
=====

- The default date formats are now all based on ISO format, i.e., with
  the slowest-moving value first.  The date formatters are still
  changeable through the ``date.autoformatter.*`` rcParams.  Python's
  ``%x`` and ``%X`` date formats may be of particular interest to
  format dates based on the current locale.

Legends
=======

- By default, the number of points displayed in a legend is now 1.

- The default legend location is ``best``, so the legend will be
  automatically placed in a location to obscure the least amount of
  data possible.

- The legend now has rounded corners by default.

mplot3d
=======

- mplot3d now obeys some style-related rcParams, rather than using
  hard-coded defaults.  These include:

  - xtick.major.width
  - ytick.major.width
  - xtick.color
  - ytick.color
  - axes.linewidth
  - axes.edgecolor
  - grid.color
  - grid.linewidth
  - grid.linestyle
