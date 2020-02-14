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

  import matplotlib.style
  import matplotlib as mpl
  mpl.style.use('classic')

See :ref:`customizing-with-matplotlibrc-files` for details about how to
persistently and selectively revert many of these changes.


.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry



Colors, color cycles, and color maps
====================================

Colors in default property cycle
--------------------------------

The colors in the default property cycle have been changed from
``['b', 'g', 'r', 'c', 'm', 'y', 'k']`` to the category10
color palette used by `Vega
<https://github.com/vega/vega/wiki/Scales#scale-range-literals>`__ and
`d3
<https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10>`__
originally developed at Tableau.


.. plot::

  import numpy as np
  import matplotlib.pyplot as plt

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

  fig.subplots_adjust(**{'bottom': 0.0, 'left': 0.059,
                         'right': 0.869, 'top': 0.895})

In addition to changing the colors, an additional method to specify
colors was added.  Previously, the default colors were the single
character short-hand notations for red, green, blue, cyan, magenta,
yellow, and black.  This made them easy to type and usable in the
abbreviated style string in ``plot``, however the new default colors
are only specified via hex values.  To access these colors outside of
the property cycling the notation for colors ``'CN'``, where ``N``
takes values 0-9, was added to
denote the first 10 colors in :rc:`axes.prop_cycle`. See
:doc:`/tutorials/colors/colors` for more details.

To restore the old color cycle use

.. code::

   from cycler import cycler
   mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

or set

.. code::

   axes.prop_cycle    : cycler('color', 'bgrcmyk')

in your :file:`matplotlibrc` file.


Colormap
--------

The new default color map used by `matplotlib.cm.ScalarMappable` instances is
'viridis' (aka `option D <https://bids.github.io/colormap/>`__).

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   N = M = 200
   X, Y = np.ogrid[0:20:N*1j, 0:20:M*1j]
   data = np.sin(np.pi * X*2 / 20) * np.cos(np.pi * Y*2 / 20)

   fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7, 3))
   im = ax1.imshow(data, extent=[0, 200, 0, 200])
   ax1.set_title("v2.0: 'viridis'")
   fig.colorbar(im, ax=ax1, shrink=0.8)

   im2 = ax2.imshow(data, extent=[0, 200, 0, 200], cmap='jet')
   fig.colorbar(im2, ax=ax2, shrink=0.8)
   ax2.set_title("classic: 'jet'")

   fig.tight_layout()

For an introduction to color theory and how 'viridis' was generated
watch Nathaniel Smith and Stéfan van der Walt's talk from SciPy2015.
See `here for many more details <https://bids.github.io/colormap/>`__
about the other alternatives and the tools used to create the color
map.  For details on all of the color maps available in matplotlib see
:doc:`/tutorials/colors/colormaps`.

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

or by setting ::


    figure.facecolor : '0.75'

in your :file:`matplotlibrc` file.


Grid lines
----------

The default style of grid lines was changed from black dashed lines to thicker
solid light grey lines.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

   ax1.grid(color='k', linewidth=.5, linestyle=':')
   ax1.set_title('classic')

   ax2.grid()
   ax2.set_title('v2.0')

The previous default can be restored by using::

   mpl.rcParams['grid.color'] = 'k'
   mpl.rcParams['grid.linestyle'] = ':'
   mpl.rcParams['grid.linewidth'] = 0.5

or by setting::

   grid.color       :   k       # grid color
   grid.linestyle   :   :       # dotted
   grid.linewidth   :   0.5     # in points

in your :file:`matplotlibrc` file.


Figure size, font size, and screen dpi
======================================

The default dpi used for on-screen display was changed from 80 dpi to
100 dpi, the same as the default dpi for saving files.  Due to this
change, the on-screen display is now more what-you-see-is-what-you-get
for saved files.  To keep the figure the same size in terms of pixels, in
order to maintain approximately the same size on the screen, the
default figure size was reduced from 8x6 inches to 6.4x4.8 inches.  As
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

or by setting::

   figure.figsize   : [8.0, 6.0]
   figure.dpi       : 80
   savefig.dpi      : 100

   font.size        : 12.0
   legend.fontsize  : 'large'
   figure.titlesize : 'medium'

In your :file:`matplotlibrc` file.

In addition, the ``forward`` kwarg to
`~matplotlib.Figure.set_size_inches` now defaults to `True` to improve
the interactive experience.  Backend canvases that adjust the size of
their bound `matplotlib.figure.Figure` must pass ``forward=False`` to
avoid circular behavior.  This default is not configurable.


Plotting functions
==================

``scatter``
-----------

The following changes were made to the default behavior of
`~matplotlib.axes.Axes.scatter`

- The default size of the elements in a scatter plot is now based on
  :rc:`lines.markersize` so it is consistent with ``plot(X,
  Y, 'o')``.  The old value was 20, and the new value is 36 (6^2).
- Scatter markers no longer have a black edge.
- If the color of the markers is not specified it will follow the
  property cycle, pulling from the 'patches' cycle on the ``Axes``.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

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

- the default linewidth increased from 1 to 1.5
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

``errorbar``
------------

By default, caps on the ends of errorbars are not present.

.. plot::

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import numpy as np

   # example data
   x = np.arange(0.1, 4, 0.5)
   y = np.exp(-x)

   # example variable error bar values
   yerr = 0.1 + 0.2*np.sqrt(x)
   xerr = 0.1 + yerr

   def demo(ax, rc, title):
       with mpl.rc_context(rc=rc):
           ax.errorbar(x, y, xerr=0.2, yerr=0.4)
       ax.set_title(title)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

   demo(ax1, {'errorbar.capsize': 3}, 'classic')
   demo(ax2, {}, 'v2.0')

This also changes the return value of
:meth:`~matplotlib.axes.Axes.errorbar` as the list of 'caplines' will
be empty by default.

The previous defaults can be restored by setting::

    mpl.rcParams['errorbar.capsize'] = 3

or by setting ::

   errorbar.capsize : 3

in your :file:`matplotlibrc` file.


``boxplot``
-----------

Previously, boxplots were composed of a mish-mash of styles that were, for
better for worse, inherited from Matlab. Most of the elements were blue,
but the medians were red. The fliers (outliers) were black plus-symbols
('+') and the whiskers were dashed lines, which created ambiguity if
the (solid and black) caps were not drawn.

For the new defaults, everything is black except for the median and mean
lines (if drawn), which are set to the first two elements of the current
color cycle. Also, the default flier markers are now hollow circles,
which maintain the ability of the plus-symbols to overlap without
obscuring data too much.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.random.lognormal(size=(37, 4))
    fig, (old, new) = plt.subplots(ncols=2, sharey=True)
    with plt.style.context('default'):
        new.boxplot(data, labels=['A', 'B', 'C', 'D'])
        new.set_title('v2.0')

    with plt.style.context('classic'):
        old.boxplot(data, labels=['A', 'B', 'C', 'D'])
        old.set_title('classic')

    new.set_yscale('log')
    old.set_yscale('log')

The previous defaults can be restored by setting::

    mpl.rcParams['boxplot.flierprops.color'] = 'k'
    mpl.rcParams['boxplot.flierprops.marker'] = '+'
    mpl.rcParams['boxplot.flierprops.markerfacecolor'] = 'none'
    mpl.rcParams['boxplot.flierprops.markeredgecolor'] = 'k'
    mpl.rcParams['boxplot.boxprops.color'] = 'b'
    mpl.rcParams['boxplot.whiskerprops.color'] = 'b'
    mpl.rcParams['boxplot.whiskerprops.linestyle'] = '--'
    mpl.rcParams['boxplot.medianprops.color'] = 'r'
    mpl.rcParams['boxplot.meanprops.color'] = 'r'
    mpl.rcParams['boxplot.meanprops.marker'] = '^'
    mpl.rcParams['boxplot.meanprops.markerfacecolor'] = 'r'
    mpl.rcParams['boxplot.meanprops.markeredgecolor'] = 'k'
    mpl.rcParams['boxplot.meanprops.markersize'] = 6
    mpl.rcParams['boxplot.meanprops.linestyle'] = '--'
    mpl.rcParams['boxplot.meanprops.linewidth'] = 1.0

or by setting::

    boxplot.flierprops.color:           'k'
    boxplot.flierprops.marker:          '+'
    boxplot.flierprops.markerfacecolor: 'none'
    boxplot.flierprops.markeredgecolor: 'k'
    boxplot.boxprops.color:             'b'
    boxplot.whiskerprops.color:         'b'
    boxplot.whiskerprops.linestyle:     '--'
    boxplot.medianprops.color:          'r'
    boxplot.meanprops.color:            'r'
    boxplot.meanprops.marker:           '^'
    boxplot.meanprops.markerfacecolor:  'r'
    boxplot.meanprops.markeredgecolor:  'k'
    boxplot.meanprops.markersize:        6
    boxplot.meanprops.linestyle:         '--'
    boxplot.meanprops.linewidth:         1.0

in your :file:`matplotlibrc` file.


``fill_between`` and ``fill_betweenx``
--------------------------------------

`~matplotlib.axes.Axes.fill_between` and
`~matplotlib.axes.Axes.fill_betweenx` both follow the patch color
cycle.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
   fig.subplots_adjust(wspace=0.3)
   th = np.linspace(0, 2*np.pi, 128)
   N = 5

   def demo(ax, extra_kwargs, title):
       ax.set_title(title)
       return [ax.fill_between(th, np.sin((j / N) * np.pi + th), alpha=.5, **extra_kwargs)
               for j in range(N)]

   demo(ax1, {'facecolor': 'C0'}, 'classic')
   demo(ax2, {}, 'v2.0')


If the facecolor is set via the ``facecolors`` or ``color`` keyword argument,
then the color is not cycled.

To restore the previous behavior, explicitly pass the keyword argument
``facecolors='C0'`` to the method call.


Patch edges and color
---------------------

Most artists drawn with a patch (``~matplotlib.axes.Axes.bar``,
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
           ax_mid.bar(range(len(fracs)), fracs, tick_label=labels)
           plt.setp(ax_mid.get_xticklabels(), rotation=-45)
           grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

           ax_bottom.set_xlim(0, .75)
           ax_bottom.set_ylim(0, .75)
           ax_bottom.add_artist(mpatches.Rectangle(grid[1] - [0.025, 0.05],
                                                   0.05, 0.1))
           ax_bottom.add_artist(mpatches.RegularPolygon(grid[3], 5, 0.1))
           ax_bottom.add_artist(mpatches.Ellipse(grid[4], 0.2, 0.1))
           ax_bottom.add_artist(mpatches.Circle(grid[0], 0.1))
           ax_bottom.axis('off')

   demo(*all_ax[:, 0], rcparams={'patch.force_edgecolor': True,
                                 'patch.facecolor': 'b'}, label='classic')
   demo(*all_ax[:, 1], rcparams={}, label='v2.0')

The previous defaults can be restored by setting::

    mpl.rcParams['patch.force_edgecolor'] = True
    mpl.rcParams['patch.facecolor'] = 'b'

or by setting::

   patch.facecolor        : b
   patch.force_edgecolor  : True

in your :file:`matplotlibrc` file.

``hexbin``
----------

The default value of the ``linecolor`` kwarg for `~matplotlib.Axes.hexbin` has
changed from ``'none'`` to ``'face'``. If 'none' is now supplied, no line edges
are drawn around the hexagons.

.. _barbarh_align:

``bar`` and ``barh``
--------------------

The default value of the ``align`` kwarg for both
`~matplotlib.Axes.bar` and `~matplotlib.Axes.barh` is changed from
``'edge'`` to ``'center'``.


.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

   def demo(bar_func, bar_kwargs):
       return bar_func([1, 2, 3], [1, 2, 3], tick_label=['a', 'b', 'c'],
                       **bar_kwargs)


   ax1.set_title("classic")
   ax2.set_title('v2.0')

   demo(ax1.bar, {'align': 'edge'})
   demo(ax2.bar, {})
   demo(ax3.barh, {'align': 'edge'})
   demo(ax4.barh, {})


To restore the previous behavior explicitly pass the keyword argument
``align='edge'`` to the method call.


Hatching
========


The color of the lines in the hatch is now determined by

- If an edge color is explicitly set, use that for the hatch color
- If the edge color is not explicitly set, use :rc:`hatch.color` which
  is looked up at artist creation time.

The width of the lines in a hatch pattern is now configurable by the
rcParams :rc:`hatch.linewidth`, which defaults to 1 point.  The old
behavior for the line width was different depending on backend:

- PDF: 0.1 pt
- SVG: 1.0 pt
- PS:  1 px
- Agg: 1 px

The old line width behavior can not be restored across all backends
simultaneously, but can be restored for a single backend by setting::

   mpl.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth
   mpl.rcParams['hatch.linewidth'] = 1.0  # previous svg hatch linewidth

The behavior of the PS and Agg backends was DPI dependent, thus::


   mpl.rcParams['figure.dpi'] = dpi
   mpl.rcParams['savefig.dpi'] = dpi  # or leave as default 'figure'
   mpl.rcParams['hatch.linewidth'] = 1.0 / dpi  # previous ps and Agg hatch linewidth


There is no direct API level control of the hatch color or linewidth.

Hatching patterns are now rendered at a consistent density, regardless of DPI.
Formerly, high DPI figures would be more dense than the default, and low DPI
figures would be less dense.  This old behavior cannot be directly restored,
but the density may be increased by repeating the hatch specifier.


.. _default_changes_font:

Fonts
=====

Normal text
-----------

The default font has changed from "Bitstream Vera Sans" to "DejaVu
Sans".  DejaVu Sans has additional international and math characters,
but otherwise has the same appearance as Bitstream Vera Sans.
Latin, Greek, Cyrillic, Armenian, Georgian, Hebrew, and Arabic are
`all supported <https://dejavu-fonts.github.io/>`__
(but right-to-left rendering is still not handled by matplotlib).
In addition, DejaVu contains a sub-set of emoji symbols.

.. plot::

   from __future__ import unicode_literals

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   tick_labels = ['😃', '😎', '😴', '😲', '😻']
   bar_labels = ['א', 'α', '☣', '⌬', 'ℝ']
   y = [1, 4, 9, 16, 25]
   x = range(5)
   ax.bar(x, y, tick_label=tick_labels, align='center')
   ax.xaxis.set_tick_params(labelsize=20)
   for _x, _y, t in zip(x, y, bar_labels):
       ax.annotate(t, (_x, _y), fontsize=20, ha='center',
                   xytext=(0, -2), textcoords='offset pixels',
                   bbox={'facecolor': 'w'})

   ax.set_title('Диаграмма со смайликами')

See the `DejaVu Sans PDF sample for full coverage
<http://dejavu.sourceforge.net/samples/DejaVuSans.pdf>`__.

Math text
---------

The default math font when using the built-in math rendering engine
(mathtext) has changed from "Computer Modern" (i.e. LaTeX-like) to
"DejaVu Sans".  This change has no effect if the
TeX backend is used (i.e. ``text.usetex`` is ``True``).


.. plot::

   import matplotlib.pyplot as plt
   import matplotlib as mpl

   mpl.rcParams['mathtext.fontset'] = 'cm'
   mpl.rcParams['mathtext.rm'] = 'serif'

   fig, ax = plt.subplots(tight_layout=True, figsize=(3, 3))

   ax.plot(range(15), label=r'int: $15 \int_0^\infty dx$')
   ax.legend()
   ax.set_title('classic')


.. plot::

   import matplotlib.pyplot as plt
   import matplotlib as mpl

   fig, ax  = plt.subplots(tight_layout=True, figsize=(3, 3))

   ax.plot(range(15), label=r'int: $15 \int_0^\infty dx$')
   ax.legend()
   ax.set_title('v2.0')



To revert to the old behavior set the::

   mpl.rcParams['mathtext.fontset'] = 'cm'
   mpl.rcParams['mathtext.rm'] = 'serif'

or set::

  mathtext.fontset: cm
  mathtext.rm : serif


in your :file:`matplotlibrc` file.

This ``rcParam`` is consulted when the text is drawn, not when the
artist is created. Thus all mathtext on a given ``canvas`` will use the
same fontset.


Legends
=======

- By default, the number of points displayed in a legend is now 1.
- The default legend location is ``'best'``, so the legend will be
  automatically placed in a location to minimize overlap with data.
- The legend defaults now include rounded corners, a lighter
  boundary, and partially transparent boundary and background.

.. plot::

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import numpy as np

   def demo(ax, rcparams, title):
       np.random.seed(2)
       N = 25
       with mpl.rc_context(rc=rcparams):
           x = range(N)
           y = np.cumsum(np.random.randn(N) )
           # unpack the single Line2D artist
           ln, = ax.plot(x, y, marker='s',
                         linestyle='-', label='plot')
           ax.fill_between(x, y, 0, label='fill', alpha=.5, color=ln.get_color())
           ax.scatter(N*np.random.rand(N), np.random.rand(N), label='scatter')
           ax.set_title(title)
           ax.legend()

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

   classic_rc = {'legend.fancybox': False,
                 'legend.numpoints': 2,
                 'legend.scatterpoints': 3,
                 'legend.framealpha': None,
                 'legend.edgecolor': 'inherit',
                 'legend.loc': 'upper right',
                 'legend.fontsize': 'large'}

   demo(ax1, classic_rc, 'classic')
   demo(ax2, {}, 'v2.0')


The previous defaults can be restored by setting::

   mpl.rcParams['legend.fancybox'] = False
   mpl.rcParams['legend.loc'] = 'upper right'
   mpl.rcParams['legend.numpoints'] = 2
   mpl.rcParams['legend.fontsize'] = 'large'
   mpl.rcParams['legend.framealpha'] = None
   mpl.rcParams['legend.scatterpoints'] = 3
   mpl.rcParams['legend.edgecolor'] = 'inherit'


or by setting::

   legend.fancybox      : False
   legend.loc           : upper right
   legend.numpoints     : 2      # the number of points in the legend line
   legend.fontsize      : large
   legend.framealpha    : None    # opacity of legend frame
   legend.scatterpoints : 3 # number of scatter points
   legend.edgecolor     : inherit   # legend edge color ('inherit'
                                    # means it uses axes.edgecolor)

in your :file:`matplotlibrc` file.

Image
=====

Interpolation
-------------

The default interpolation method for `~matplotlib.axes.Axes.imshow` is
now ``'nearest'`` and by default it resamples the data (both up and down
sampling) before color mapping.


.. plot::

   import matplotlib.pyplot as plt
   import matplotlib as mpl
   import numpy as np


   def demo(ax, rcparams, title):
       np.random.seed(2)
       A = np.random.rand(5, 5)

       with mpl.rc_context(rc=rcparams):
           ax.imshow(A)
           ax.set_title(title)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

   classic_rcparams = {'image.interpolation': 'bilinear',
                       'image.resample': False}

   demo(ax1, classic_rcparams, 'classic')
   demo(ax2, {}, 'v2.0')


To restore the previous behavior set::

   mpl.rcParams['image.interpolation'] = 'bilinear'
   mpl.rcParams['image.resample'] = False

or set::

  image.interpolation  : bilinear  # see help(imshow) for options
  image.resample  : False

in your :file:`matplotlibrc` file.

Colormapping pipeline
---------------------

Previously, the input data was normalized, then color mapped, and then
resampled to the resolution required for the screen.  This meant that
the final resampling was being done in color space.  Because the color
maps are not generally linear in RGB space, colors not in the color map
may appear in the final image.  This bug was addressed by an almost
complete overhaul of the image handling code.

The input data is now normalized, then resampled to the correct
resolution (in normalized dataspace), and then color mapped to
RGB space.  This ensures that only colors from the color map appear
in the final image. (If your viewer subsequently resamples the image,
the artifact may reappear.)

The previous behavior cannot be restored.


Shading
-------

- The default shading mode for light source shading, in
  ``matplotlib.colors.LightSource.shade``, is now ``overlay``.
  Formerly, it was ``hsv``.


Plot layout
===========

Auto limits
-----------

The previous auto-scaling behavior was to find 'nice' round numbers
as view limits that enclosed the data limits, but this could produce
bad plots if the data happened to fall on a vertical or
horizontal line near the chosen 'round number' limit.  The new default
sets the view limits to 5% wider than the data range.

.. plot::

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import numpy

   data = np.zeros(1000)
   data[0] = 1

   fig = plt.figure(figsize=(6, 3))

   def demo(fig, rc, title, j):
       with mpl.rc_context(rc=rc):
           ax = fig.add_subplot(1, 2, j)
           ax.plot(data)
           ax.set_title(title)

   demo(fig, {'axes.autolimit_mode': 'round_numbers',
              'axes.xmargin': 0,
              'axes.ymargin': 0}, 'classic', 1)
   demo(fig, {}, 'v2.0', 2)

The size of the padding in the x and y directions is controlled by the
``'axes.xmargin'`` and ``'axes.ymargin'`` rcParams respectively. Whether
the view limits should be 'round numbers' is controlled by
:rc:`axes.autolimit_mode`.  In the original ``'round_number'`` mode,
the view limits coincide with ticks.

The previous default can be restored by using::

   mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
   mpl.rcParams['axes.xmargin'] = 0
   mpl.rcParams['axes.ymargin'] = 0

or setting::

   axes.autolimit_mode: round_numbers
   axes.xmargin: 0
   axes.ymargin: 0

in your :file:`matplotlibrc` file.


Z-order
-------

- Ticks and grids are now plotted above solid elements such as
  filled contours, but below lines.  To return to the previous
  behavior of plotting ticks and grids above lines, set
  ``rcParams['axes.axisbelow'] = False``.


Ticks
-----

Direction
~~~~~~~~~

To reduce the collision of tick marks with data, the default ticks now
point outward by default.  In addition, ticks are now drawn only on
the bottom and left spines to prevent a porcupine appearance, and for
a cleaner separation between subplots.


.. plot::

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import numpy as np

   th = np.linspace(0, 2*np.pi, 128)
   y = np.sin(th)

   def demo(fig, rcparams, title, j):
       np.random.seed(2)
       with mpl.rc_context(rc=rcparams):

           ax = fig.add_subplot(2, 2, j)
           ax.hist(np.random.beta(0.5, 0.5, 10000), 25, density=True)
           ax.set_xlim([0, 1])
           ax.set_title(title)

           ax = fig.add_subplot(2, 2, j + 2)
           ax.imshow(np.random.rand(5, 5))

   classic = {'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True}

   fig = plt.figure(figsize=(6, 6), tight_layout=True)

   demo(fig, classic, 'classic', 1)
   demo(fig, {}, 'v2.0', 2)


To restore the previous behavior set::

   mpl.rcParams['xtick.direction'] = 'in'
   mpl.rcParams['ytick.direction'] = 'in'
   mpl.rcParams['xtick.top'] = True
   mpl.rcParams['ytick.right'] = True

or set::

   xtick.top: True
   xtick.direction: in

   ytick.right: True
   ytick.direction: in

in your :file:`matplotlibrc` file.



Number of ticks
~~~~~~~~~~~~~~~

The default `~matplotlib.ticker.Locator` used for the x and y axis is
`~matplotlib.ticker.AutoLocator` which tries to find, up to some
maximum number, 'nicely' spaced ticks.  The locator now includes
an algorithm to estimate the maximum number of ticks that will leave
room for the tick labels.  By default it also ensures that there are at least
two ticks visible.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   from matplotlib.ticker import AutoLocator

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3), tight_layout=True)
   ax1.set_xlim(0, .1)
   ax2.set_xlim(0, .1)

   ax1.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
   ax1.set_title('classic')
   ax2.set_title('v2.0')

There is no way, other than using ``mpl.style.use('classic')``, to restore the
previous behavior as the default.  On an axis-by-axis basis you may either
control the existing locator via: ::

  ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

or create a new `~matplotlib.ticker.MaxNLocator`::

  import matplotlib.ticker as mticker
  ax.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10])

The algorithm used by `~matplotlib.ticker.MaxNLocator` has been
improved, and this may change the choice of tick locations in some
cases.  This also affects `~matplotlib.ticker.AutoLocator`, which
uses ``MaxNLocator`` internally.

For a log-scaled axis the default locator is the
`~matplotlib.ticker.LogLocator`.  Previously the maximum number
of ticks was set to 15, and could not be changed. Now there is a
*numticks* kwarg for setting the maximum to any integer value,
to the string 'auto', or to its default value of None which is
equivalent to 'auto'.  With the 'auto' setting the maximum number
will be no larger than 9, and will be reduced depending on the
length of the axis in units of the tick font size.  As in the
case of the AutoLocator, the heuristic algorithm reduces the
incidence of overlapping tick labels but does not prevent it.


Tick label formatting
---------------------

``LogFormatter`` labeling of minor ticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minor ticks on a log axis are now labeled when the axis view limits
span a range less than or equal to the interval between two major
ticks.  See `~matplotlib.ticker.LogFormatter` for details. The
minor tick labeling is turned off when using ``mpl.style.use('classic')``,
but cannot be controlled independently via `.rcParams`.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   np.random.seed(2)

   fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
   fig.subplots_adjust(wspace=0.35, left=0.09, right=0.95)

   x = np.linspace(0.9, 1.7, 10)
   y = 10 ** x[np.random.randint(0, 10, 10)]

   ax2.semilogy(x, y)
   ax2.set_title('v2.0')

   with plt.style.context('classic'):
       ax1.semilogy(x, y)
       ax1.set_xlim(ax2.get_xlim())
       ax1.set_ylim(ax2.get_ylim())
       ax1.set_title('classic')


``ScalarFormatter`` tick label formatting with offsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the default :rc:`axes.formatter.useoffset`,
an offset will be used when it will save 4 or more digits.  This can
be controlled with the new :rc:`axes.formatter.offset_threshold`.
To restore the previous behavior of using an offset to save 2 or more
digits, use ``rcParams['axes.formatter.offset_threshold'] = 2``.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   np.random.seed(5)

   fig = plt.figure(figsize=(6, 3))
   fig.subplots_adjust(bottom=0.15, wspace=0.3, left=0.09, right=0.95)

   x = np.linspace(2000, 2008, 9)
   y = np.random.randn(9) + 50000

   with plt.rc_context(rc={'axes.formatter.offset_threshold' : 2}):
       ax1 = fig.add_subplot(1, 2, 1)
       ax1.plot(x, y)
       ax1.set_title('classic')

   ax2 = fig.add_subplot(1, 2, 2)
   ax2.plot(x, y)
   ax2.set_title('v2.0')


``AutoDateFormatter`` format strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default date formats are now all based on ISO format, i.e., with
the slowest-moving value first.  The date formatters are
configurable through the ``date.autoformatter.*`` rcParams.


+--------------------------------------+--------------------------------------+-------------------+-------------------+
| Threshold (tick interval >= than)    | rcParam                              | classic           | v2.0              |
+======================================+======================================+===================+===================+
| 365 days                             | ``'date.autoformatter.year'``        | ``'%Y'``          | ``'%Y'``          |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 30 days                              | ``'date.autoformatter.month'``       | ``'%b %Y'``       | ``'%Y-%m'``       |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 1 day                                | ``'date.autoformatter.day'``         | ``'%b %d %Y'``    | ``'%Y-%m-%d'``    |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 1 hour                               | ``'date.autoformatter.hour'``        | ``'%H:%M:%S'``    | ``'%H:%M'``       |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 1 minute                             | ``'date.autoformatter.minute'``      | ``'%H:%M:%S.%f'`` | ``'%H:%M:%S'``    |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 1 second                             | ``'date.autoformatter.second'``      | ``'%H:%M:%S.%f'`` | ``'%H:%M:%S'``    |
+--------------------------------------+--------------------------------------+-------------------+-------------------+
| 1  microsecond                       | ``'date.autoformatter.microsecond'`` | ``'%H:%M:%S.%f'`` | ``'%H:%M:%S.%f'`` |
+--------------------------------------+--------------------------------------+-------------------+-------------------+



Python's ``%x`` and ``%X`` date formats may be of particular interest
to format dates based on the current locale.

The previous default can be restored by::

   mpl.rcParams['date.autoformatter.year'] = '%Y'
   mpl.rcParams['date.autoformatter.month'] = '%b %Y'
   mpl.rcParams['date.autoformatter.day'] = '%b %d %Y'
   mpl.rcParams['date.autoformatter.hour'] = '%H:%M:%S'
   mpl.rcParams['date.autoformatter.minute'] = '%H:%M:%S.%f'
   mpl.rcParams['date.autoformatter.second'] = '%H:%M:%S.%f'
   mpl.rcParams['date.autoformatter.microsecond'] = '%H:%M:%S.%f'


or setting ::

   date.autoformatter.year   : %Y
   date.autoformatter.month  : %b %Y
   date.autoformatter.day    : %b %d %Y
   date.autoformatter.hour   : %H:%M:%S
   date.autoformatter.minute : %H:%M:%S.%f
   date.autoformatter.second : %H:%M:%S.%f
   date.autoformatter.microsecond : %H:%M:%S.%f

in your :file:`matplotlibrc` file.

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
