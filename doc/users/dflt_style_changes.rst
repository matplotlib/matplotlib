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

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry



colors, color cycles, and color maps
====================================

Colors in default property cycle
--------------------------------

The colors in the default proprety cycle have been changed from
``['b', 'g', 'r', 'c', 'm', 'y', 'k']`` to the `Vega category10 palette
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
abbreviated style string in ``plot``, however the new default colors
are only specified via hex values.  To access these colors outside of
the property cycling the notation for colors ``'CN'`` was added to
denote the first 10 colors in ``mpl.rcParams['axes.prop_cycle']`` See
:ref:`colors` for more details.

To restore the old color cycle use

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
 `'viridis'` (aka `option D <https://bids.github.io/colormap/>`__).

.. plot::

   import numpy as np
   N = M = 200
   X, Y = np.ogrid[0:20:N*1j, 0:20:M*1j]
   data = np.sin(np.pi * X*2 / 20) * np.cos(np.pi * Y*2 / 20)

   fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7, 3))
   im = ax1.imshow(data, extent=[0, 200, 0, 200])
   ax1.set_title("v2.0: 'viridis'")
   fig.colorbar(im, ax=ax1, shrink=.9)

   im2 = ax2.imshow(data, extent=[0, 200, 0, 200], cmap='jet')
   fig.colorbar(im2, ax=ax2, shrink=.9)
   ax2.set_title("classic: 'jet'")

   fig.tight_layout()

For an introduction to color theory and how 'viridis' was generated
watch Nathaniel Smith and Stéfan van der Walt's talk from SciPy2015.
See `here for many more details <https://bids.github.io/colormap/>`__
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

in your :file:`matplotlibrc` file; however this is strongly discouraged.

Interactive figures
-------------------

The default interactive figure background color has changed from grey
to white, which matches the default background color used when saving.

The previous defaults can be restored by ::

   mpl.rcParams['figure.facecolor'] = '0.75'

or setting ::


    figure.facecolor : '0.75'

in your :file:`matplotlibrc` file.


Grid lines
----------

The default style of grid lines was changed from, black dashed lines to thicker
solid light grey lines.

.. plot::

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

   ax1.grid(color='k', linewidth=.5, linestyle=':')
   ax1.set_title('classic')

   ax2.grid()
   ax2.set_title('v2.0')

The previous default can be restored by using::

   mpl.rcParams['grid.color'] = 'k'
   mpl.rcParams['grid.linestyle'] = ':'
   mpl.rcParams['grid.linewidth'] = 0.5

or setting::

   grid.color       :   k       # grid color
   grid.linestyle   :   :       # dotted
   grid.linewidth   :   0.5     # in points

in your :file:`matplotlibrc` file.


Figure size, font size, and screen dpi
======================================

The default dpi used for on-screen display was changed from 80dpi to
100dpi, the same as the default dpi for saving files.  Due to this
change, the on-screen display is now more what-you-see-is-what-you-get
for saved files.  To keep figure the same size in terms of pixels, in
order to maintain approximately the same size on the screen, the
default figure size was reduced from 8in by 6in to 6.4in by 4.8in.  As
a consequence of this the default font sizes used for the title, tick
labels, and axes labels were reduced to maintain their size relative
to the overall size of the figure.  By default the dpi of the saved
image is now the dpi of the `~matplotlib.figure.Figure` instance being
saved.

This will have consequences if you are trying to match text in a
figure directly with external text.


The previous defaults can be restored by ::

   mpl.rcParams['figure.figsize'] = [8.0, 6.0]
   mpl.rcParams['figure.dpi'] = 80
   mpl.rcParams['savefig.dpi'] = 100

   mpl.rcParams['font.size'] = 12
   mpl.rcParams['legend.fontsize'] = 'large'
   mpl.rcParams['figure.titlesize'] = 'medium'

or set::

   figure.figsize   : [8.0, 6.0]
   figure.dpi       : 80
   savefig.dpi      : 100

   font.size        : 12.0
   legend.fontsize  : 'large'
   figure.titlesize : 'medium'

In your :file:`matplotlibrc` file.


Plotting functions
==================

``scatter``
-----------

The following changes were made to the default behavior of `~matplotlib.axes.Axes.scatter`

 - The default size of the elements in a scatter plot is now based on
   the rcParam ``lines.markersize`` so it is consistent with ``plot(X,
   Y, 'o')``.  The old value was 20, and the new value is 36 (6^2).
 - scatter markers no longer have a black edge.
 - if the color of the markers is not specified it will follow the property cycle
   pulling from the 'patches' cycle on the ``Axes``.

.. plot::

   np.random.seed(2)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

   x = np.arange(15)
   y = np.random.rand(15)
   y2 = np.random.rand(15)
   ax1.scatter(x, y, s=20, edgecolors='k', c='b', label='a')
   ax1.scatter(x, y2, s=20, edgecolors='k', c='b', label='b')
   ax1.legend()
   ax1.set_title('classic')

   ax2.scatter(x, y, label='a')
   ax2.scatter(x, y2, label='b')
   ax2.legend()
   ax2.set_title('v2.0')


The classic default behavior of `~matplotlib.axes.Axes.scatter` can
only be recovered through ``mpl.style.use('classic')``.  The marker size
can be recovered via ::

  mpl.rcParam['lines.markersize'] = np.sqrt(20)

however, this will also affect the default marker size of
`~matplotlib.axes.Axes.plot`.  To recover the classic behavior on
a per-call basis pass the following kwargs::

  classic_kwargs = {'s': 20, 'edgecolors': 'k', 'c': 'b'}

``plot``
--------

The following changes were made to the default behavior of
`~matplotlib.axes.Axes.plot`

 - the default linewidth change from 1 to 1.5
 - the dash patterns associated with ``'--'``, ``':'``, and ``'-.'`` have
   changed
 - the dash patterns now scale with line width


.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   from cycler import cycler

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

   N = 15

   x = np.arange(N)
   y = np.ones_like(x)

   sty_cycle = (cycler('ls', ['--' ,':', '-.']) *
                cycler('lw', [None, 1, 2, 5]))

   classic = {
       'lines.linewidth': 1.0,
       'lines.dashed_pattern' : [6, 6],
       'lines.dashdot_pattern' : [3, 5, 1, 5],
       'lines.dotted_pattern' : [1, 3],
       'lines.scale_dashes': False}

   v2 = {}
   #    {'lines.linewidth': 1.5,
   #     'lines.dashed_pattern' : [2.8, 1.2],
   #     'lines.dashdot_pattern' : [4.8, 1.2, 0.8, 1.2],
   #     'lines.dotted_pattern' : [1.1, 1.1],
   #     'lines.scale_dashes': True}

   def demo(ax, rcparams, title):
       ax.axis('off')
       ax.set_title(title)
       with mpl.rc_context(rc=rcparams):
           for j, sty in enumerate(sty_cycle):
               ax.plot(x, y + j, **sty)

   demo(ax1, classic, 'classic')
   demo(ax2, {}, 'v2.0')


The previous defaults can be restored by setting::

    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.dashed_pattern'] = [6, 6]
    mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
    mpl.rcParams['lines.dotted_pattern'] = [1, 3]
    mpl.rcParams['lines.scale_dashes'] = False

or by setting::

   lines.linewidth   : 1.0
   lines.dashed_pattern : 6, 6
   lines.dashdot_pattern : 3, 5, 1, 5
   lines.dotted_pattern : 1, 3
   lines.scale_dashes: False

in your :file:`matplotlibrc` file.

Patch edges and color
---------------------

Artists drawn with a patch (``~matplotlib.axes.Axes.bar``,
``~matplotlib.axes.Axes.pie``, etc) no longer have a black edge by
default.  The default face color is now ``'C0'`` instead of ``'b'``.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib import rc_context
   import matplotlib.patches as mpatches

   fig, all_ax = plt.subplots(3, 2, figsize=(4, 6), tight_layout=True)

   def demo(ax_top, ax_mid, ax_bottom, rcparams, label):
       labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
       fracs = [15, 30, 45, 10]

       explode = (0, 0.05, 0, 0)

       ax_top.set_title(label)

       with rc_context(rc=rcparams):
           ax_top.pie(fracs, labels=labels)
           ax_top.set_aspect('equal')
           ax_mid.bar(range(len(fracs)), fracs, tick_label=labels, align='center')
           plt.setp(ax_mid.get_xticklabels(), rotation=-45)
           grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

           ax_bottom.set_xlim(0, .75)
           ax_bottom.set_ylim(0, .75)
           ax_bottom.add_artist(mpatches.Rectangle(grid[1] - [0.025, 0.05], 0.05, 0.1))
           ax_bottom.add_artist(mpatches.RegularPolygon(grid[3], 5, 0.1))
           ax_bottom.add_artist(mpatches.Ellipse(grid[4], 0.2, 0.1))
           ax_bottom.add_artist(mpatches.Circle(grid[0], 0.1))
           ax_bottom.axis('off')

   demo(*all_ax[:, 0], rcparams={'patch.force_edgecolor': True,
                                 'patch.facecolor': 'b'}, label='classic')
   demo(*all_ax[:, 1], rcparams={}, label='v2.0')

The previous defaults can be restored by setting::

    mpl.rcParams['patch.force_edgecolor'] = True
    mpl.rcParams['patch.facecolor'] = True

or by setting::

   patch.facecolor        : b
   patch.force_edgecolor  : True

in your :file:`matplotlibrc` file.



Other
=====

- lines.color change, only hits raw usage of Line2D

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

- The number of ticks on an axis is now automatically determined based
  on the length of the axis.

- The limits of an axes are scaled to exactly the dimensions of the data,
  plus 5% padding.  The old behavior was to scale to the nearest "round"
  numbers.  To use the old behavior, set the ``rcParam``
  ``axes.autolimit_mode`` to ``round_numbers``.  To control the
  margins on a particular side individually, pass any of the following
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
