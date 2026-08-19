==============================================
What's new in Matplotlib 3.11.0 (Jun 11, 2026)
==============================================

For a list of all of the issues and pull requests since the last revision, see the
:ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 4

.. toctree::
   :maxdepth: 4

Figure creation / management
============================

Figures can be attached to and removed from pyplot
--------------------------------------------------

Figures can now be attached to and removed from management through pyplot, which in the
background also means a less strict coupling to backends.

In particular, standalone figures (created with the `.Figure` constructor) can now be
registered with the `.pyplot` module by calling ``plt.figure(fig)``. This allows showing
them with ``plt.show()`` as you would do with any figure created with pyplot factory
functions such as ``plt.figure()`` or ``plt.subplots()``.

When closing a shown figure window, the related figure is reset to the standalone state,
i.e., it's not visible to pyplot anymore, but if you still hold a reference to it, you
can continue to work with it (e.g. do ``fig.savefig()``, or re-add it to pyplot with
``plt.figure(fig)`` and then show it again).

The following is now possible — though the example is exaggerated to show what's
possible. In practice, you'll stick with much simpler versions for better consistency ::

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # Create a standalone figure
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([1, 2, 3], [4, 5, 6])

    # Register it with pyplot
    plt.figure(fig)

    # Modify the figure through pyplot
    plt.xlabel("x label")

    # Show the figure
    plt.show()

    # Close the figure window through the GUI

    # Continue to work on the figure
    fig.savefig("my_figure.png")
    ax.set_ylabel("y label")

    # Re-register the figure and show it again
    plt.figure(fig)
    plt.show()

.. dropdown:: Technical detail
    :color: info
    :icon: info

    Standalone figures use `.FigureCanvasBase` as canvas. This is replaced by a
    backend-dependent subclass when registering with pyplot, and is reset to
    `.FigureCanvasBase` when the figure is closed. `.Figure.savefig` uses the current
    canvas to save the figure (if possible). Since `.FigureCanvasBase` can not render
    the figure, when saving the figure, it will fall back to a suitable canvas subclass,
    e.g., `.FigureCanvasAgg` for raster outputs such as PNG.

    Any Agg-based backend will create the same file output. However, there may be slight
    differences for non-Agg backends; e.g. if you use "GTK4Cairo" as interactive
    backend, ``fig.savefig("file.png")`` may create a slightly different image depending
    on whether the figure is registered with pyplot or not.

    In general, you should not store a reference to the canvas, but rather always obtain
    it from the figure with ``fig.canvas``. This will return the current canvas, which
    is either the original `.FigureCanvasBase` or a backend-dependent subclass,
    depending on whether the figure is registered with pyplot or not.

Figure size units
-----------------

When creating figures, it is now possible to define figure sizes in centimetres or
pixels.

Up to now the figure size is specified via ``plt.figure(..., figsize=(6, 4))``, and the
given numbers are interpreted as inches. It is now possible to add a unit string to the
tuple, i.e. ``plt.figure(..., figsize=(600, 400, "px"))``. Supported unit strings are
``"in"``, ``"cm"``, or ``"px"``.

Partial ``figsize`` specification at figure creation
----------------------------------------------------

Figure creation now accepts a single ``None`` in ``figsize``. Passing ``(None, h)`` uses
the default width from :rc:`figure.figsize`, and passing ``(w, None)`` uses the default
height. Passing ``(None, None)`` is invalid and raises a `ValueError`.

For example::

    plt.rcParams['figure.figsize'] = (14, 11)
    fig = plt.figure(figsize=(None, 4))  # Size will be (14, 4)

Subplot parameters are reset in ``Figure.clear``
------------------------------------------------

When calling `.Figure.clear()` the settings for `.gridspec.SubplotParams` are restored
to the default values.

`.SubplotParams.to_dict` is a new method to get the subplot parameters as a dict, and
`.SubplotParams.reset` resets the parameters to the defaults.

Plotting methods
================

Grouped bar charts
------------------

The new method `~.Axes.grouped_bar()` simplifies the creation of grouped bar charts
significantly. It supports different input data types (lists of datasets, dicts of
datasets, data in 2D arrays, pandas DataFrames), and allows for easy customization of
placement via controllable distances between bars and between bar groups.

.. plot::
    :include-source:
    :alt: Diagram of a grouped bar chart of 3 datasets with 2 categories.

    categories = ['A', 'B']
    datasets = {
        'dataset 0': [1, 11],
        'dataset 1': [3, 13],
        'dataset 2': [5, 15],
    }

    fig, ax = plt.subplots()
    ax.grouped_bar(datasets, tick_labels=categories)
    ax.legend()

``broken_barh()`` vertical alignment through *align* parameter
--------------------------------------------------------------

`~.Axes.broken_barh` now supports vertical alignment of the bars through the *align*
parameter.

.. plot::
    :include-source:
    :alt:
        A plot with three horizontal bars at 0, 10, and 20. Each is aligned to the
        bottom, center, and top of those values, respectively.

    fig, ax = plt.subplots()
    ax.axhline(0, color='tab:red')
    ax.broken_barh([(0, 10)], (0, 2))  # Default is 'bottom'.
    ax.axhline(10, color='tab:red')
    ax.broken_barh([(0, 10)], (10, 2), align='center')
    ax.axhline(20, color='tab:red')
    ax.broken_barh([(0, 10)], (20, 2), align='top')

``hist()`` supports a single color for multiple datasets
--------------------------------------------------------

It is now possible to pass a single *color* value to `~.Axes.hist()`. This value is
applied to all datasets.

Stackplot styling
-----------------

`~.Axes.stackplot` now accepts sequences for the style parameters *facecolor*,
*edgecolor*, *linestyle*, and *linewidth*, similar to how the *hatch* parameter is
already handled.

.. plot::
    :include-source:
    :alt: A plot of stacked datasets. The bottom area is orange and the top is green.

    x = np.linspace(0, 10)
    y1 = x + np.sin(x)
    y2 = x + np.cos(x)

    fig, ax = plt.subplots()
    ax.stackplot(x, y1, y2, facecolor=['tab:orange', 'tab:green'])

Streamplot integration control
------------------------------

Two new options have been added to the `~.axes.Axes.streamplot` method that give better
control of the streamline integration:

``integration_max_step_scale``
    Multiplies the default max step computed by the integrator.
``integration_max_error_scale``
    Multiplies the default max error set by the integrator.

Values for these parameters between zero and one reduce (tighten) the max step or error
to improve streamline accuracy by performing more computation. Values greater than one
increase (loosen) the max step or error to reduce computation time at the cost of lower
streamline accuracy.

The integrator defaults are both hand-tuned values and may not be applicable to all
cases, so this allows customizing the behavior to specific use cases. Modifying only
``integration_max_step_scale`` has proved effective, but it may be useful to control the
error as well.

Multiple arrows on a streamline
-------------------------------

A new *num_arrows* argument has been added to `~matplotlib.axes.Axes.streamplot` that
allows more than one arrow to be added to each streamline:

.. plot::
    :include-source:
    :alt: One chart showing a streamplot. Each streamline has three arrows.

    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2

    fig, ax = plt.subplots()
    ax.streamplot(X, Y, U, V, num_arrows=3)

``violinplot`` now accepts color arguments
------------------------------------------

`~.Axes.violinplot` and `~.Axes.violin` now accept ``facecolor`` and ``linecolor`` as
input arguments. This means that the color of violinplots can be set as they are made,
rather than setting the color of individual objects afterwards. It is possible to pass a
single color to be used for all violins, or pass a sequence of colors.

.. plot::
    :include-source:
    :alt:
        Two violin plots. On the left, all elements are blue. On the right, each is a
        custom colour: a desaturated yellow, blue, red, and green for each data set,
        and black for the vertical bars.

    np.random.seed(19680801)
    data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)

    ax1.set_title('Default violin plot')
    ax1.set_ylabel('Observed values')
    ax1.violinplot(data)

    ax2.set_title('Set colors of violins')
    ax2.set_ylabel('Observed values')
    ax2.violinplot(
        data,
        facecolor=[('yellow', 0.3), ('blue', 0.3), ('red', 0.3), ('green', 0.3)],
        linecolor='black',
    )

Annotations
===========

``bar_label`` supports individual padding per label
---------------------------------------------------

`~.Axes.bar_label` will now accept both a float value or an array-like for padding. The
array-like defines the padding for each label individually.

Adding labels to pie chart wedges
---------------------------------

The new `~.Axes.pie_label` method adds a label to each wedge in a pie chart created with
`~.Axes.pie`.  It can take

* a list of strings, similar to the existing *labels* parameter of `~.Axes.pie`
* a format string similar to the existing *autopct* parameter of `~.Axes.pie` except
  that it uses the `str.format` method and it can handle absolute values as well as
  fractions/percentages

For more examples, see :doc:`/gallery/pie_and_polar_charts/pie_label`.

.. plot::
    :include-source:
    :alt:
        A pie chart with three labels on each wedge, showing a food type, number, and
        fraction associated with the wedge.

    data = [36, 24, 8, 12]
    labels = ['spam', 'eggs', 'bacon', 'sausage']

    fig, ax = plt.subplots()
    pie = ax.pie(data)

    ax.pie_label(pie, labels, distance=1.1)
    ax.pie_label(pie, '{frac:.1%}', distance=0.7)
    ax.pie_label(pie, '{absval:d}', distance=0.4)

Arrow-style sub-classes of ``BoxStyle`` support arrow head resizing
-------------------------------------------------------------------

The new *head_width* and *head_angle* parameters to `.BoxStyle.LArrow`,
`.BoxStyle.RArrow` and `.BoxStyle.DArrow` allow for adjustment of the size and aspect
ratio of the arrow heads used.

To give a consistent appearance across all parameter values, the default head position
(where the head starts relative to text) is slightly changed compared to the previous
hard-coded position.

By using negative angles (or corresponding reflex angles) for *head_angle*, arrows with
'backwards' heads may be created.

.. plot::
    :include-source:
    :alt:
        Six arrow-shaped text boxes.  The arrows on the left have the shaft on their
        left; the arrows on the right have the shaft on the right; the arrows in the
        middle have shafts on both sides.

    plt.text(0.2, 0.8, "LArrow", ha='center', size=16,
             bbox=dict(boxstyle="larrow, pad=0.3, head_angle=150"))
    plt.text(0.2, 0.2, "LArrow", ha='center', size=16,
             bbox=dict(boxstyle="larrow, pad=0.3, head_width=0.5"))
    plt.text(0.5, 0.8, "DArrow", ha='center', size=16,
             bbox=dict(boxstyle="darrow, pad=0.3, head_width=3"))
    plt.text(0.5, 0.2, "DArrow", ha='center', size=16,
             bbox=dict(boxstyle="darrow, pad=0.3, head_width=1, head_angle=60"))
    plt.text(0.8, 0.8, "RArrow", ha='center', size=16,
             bbox=dict(boxstyle="rarrow, pad=0.3, head_angle=30"))
    plt.text(0.8, 0.2, "RArrow", ha='center', size=16,
             bbox=dict(boxstyle="rarrow, pad=0.3, head_width=2, head_angle=-90"))
    plt.axis("off")

*borderpad* accepts a tuple for separate x/y padding
----------------------------------------------------

The *borderpad* parameter used for placing anchored artists (such as inset axes) now
accepts a tuple of ``(x_pad, y_pad)``.

This allows for specifying separate padding values for the horizontal and vertical
directions, providing finer control over placement. For example, when placing an inset
in a corner, one might want horizontal padding to avoid overlapping with the main plot's
axis labels, but no vertical padding to keep the inset flush with the plot area edge.

Example usage with :func:`~mpl_toolkits.axes_grid1.inset_locator.inset_axes`:

.. code-block:: python

    ax_inset = inset_axes(
        ax, width="30%", height="30%", loc='upper left',
        borderpad=(4, 0))

Axes and Artists
================

Twin Axes ``delta_zorder``
--------------------------

`~matplotlib.axes.Axes.twinx` and `~matplotlib.axes.Axes.twiny` now accept a
*delta_zorder* keyword argument, a relative offset added to the original Axes' zorder,
to control whether the twin Axes is drawn in front of, or behind, the original Axes. For
example, pass ``delta_zorder=-1`` to draw a twin Axes behind the main Axes.

In addition, Matplotlib now automatically manages background patch visibility for each
group of twinned Axes so that only the bottom-most Axes in the group has a visible
background patch (respecting ``frameon``).

``BarContainer`` properties
---------------------------

`.BarContainer` gained new properties to easily access coordinates of the bars:

- `~.BarContainer.bottoms`
- `~.BarContainer.tops`
- `~.BarContainer.position_centers`

Maximum levels on log-scaled contour plots are now respected
------------------------------------------------------------

When plotting contours with a log norm, passing an integer value to the ``levels``
argument to cap the maximum number of contour levels now works as intended.

``edgegapcolor`` for Patches
----------------------------

`~matplotlib.patches.Patch` now supports an *edgegapcolor* parameter, similar to the
existing *gapcolor* in `.Line2D`. This allows patches with dashed edges to display a
secondary color in the gaps, creating a "striped" edge effect.

This is useful when drawing unfilled patches on backgrounds of unknown color, where
alternating edge colors ensure the patch boundary remains visible.

.. plot::
    :include-source:
    :alt:
        A rectangle with a dashed orange edge and blue gaps, demonstrating the
        edgegapcolor feature.

    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()
    rect = Rectangle((0.1, 0.1), 0.6, 0.6, fill=False,
                      edgecolor='orange', edgegapcolor='blue',
                      linestyle='--', linewidth=3)
    ax.add_patch(rect)

Separated ``hatchcolor`` from ``edgecolor``
-------------------------------------------

When the *hatchcolor* parameter is specified, it will be used for the hatch. If it is
not specified, it will fall back to using :rc:`hatch.color`. The special value 'edge'
uses the patch edgecolor, with a fallback to :rc:`patch.edgecolor` if the patch
edgecolor is 'none'. Previously, hatch colors were the same as edge colors, with a
fallback to :rc:`hatch.color` if the patch did not have an edge color.

.. plot::
    :include-source:
    :alt:
        Four Rectangle patches, each displaying the color of hatches in different
        specifications of edgecolor and hatchcolor. Top left has hatchcolor='black'
        representing the default value when both hatchcolor and edgecolor are not set,
        top right has edgecolor='blue' and hatchcolor='black' which remains when the
        edgecolor is set again, bottom left has edgecolor='red' and hatchcolor='orange'
        on explicit specification and bottom right has edgecolor='green' and
        hatchcolor='green' when the hatchcolor is not set.

    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    # In this case, hatchcolor is orange
    patch1 = Rectangle((0.1, 0.1), 0.3, 0.3, edgecolor='red', linewidth=2,
                       hatch='//', hatchcolor='orange')
    ax.add_patch(patch1)

    # When hatchcolor is not specified, it matches edgecolor
    # In this case, hatchcolor is green
    patch2 = Rectangle((0.6, 0.1), 0.3, 0.3, edgecolor='green', linewidth=2,
                       hatch='//', facecolor='none')
    ax.add_patch(patch2)

    # If both hatchcolor and edgecolor are not specified
    # it will default to the 'patch.edgecolor' rcParam, which is black by default
    # In this case, hatchcolor is black
    patch3 = Rectangle((0.1, 0.6), 0.3, 0.3, hatch='//')
    ax.add_patch(patch3)

    # When using `hatch.color` in the `rcParams`
    # edgecolor will now not overwrite hatchcolor
    # In this case, hatchcolor is black
    with plt.rc_context({'hatch.color': 'black'}):
        patch4 = Rectangle((0.6, 0.6), 0.3, 0.3, edgecolor='blue', linewidth=2,
                           hatch='//', facecolor='none')

    # hatchcolor is black (it uses the `hatch.color` rcParam value)
    patch4.set_edgecolor('blue')
    # hatchcolor is still black (here, it does not update when edgecolor changes)
    ax.add_patch(patch4)

    ax.annotate("hatchcolor = 'orange'",
                xy=(.5, 1.03), xycoords=patch1, ha='center', va='bottom')
    ax.annotate("hatch color unspecified\nedgecolor='green'",
                xy=(.5, 1.03), xycoords=patch2, ha='center', va='bottom')
    ax.annotate("hatch color unspecified\nusing patch.edgecolor",
                xy=(.5, 1.03), xycoords=patch3, ha='center', va='bottom')
    ax.annotate("hatch.color='black'",
                xy=(.5, 1.03), xycoords=patch4, ha='center', va='bottom')

For collections, a sequence of colors can be passed to the *hatchcolor* parameter which
will be cycled through for each hatch, similar to *facecolor* and *edgecolor*.

Previously, if *edgecolor* was not specified, the hatch color would fall back to
:rc:`patch.edgecolor`, but the alpha value would default to **1.0**, regardless of the
alpha value of the collection. This behavior has been changed such that, if both
*hatchcolor* and *edgecolor* are not specified, the hatch color will fall back to
'patch.edgecolor' with the alpha value of the collection.

.. plot::
    :include-source:
    :alt:
        A random scatter plot with hatches on the markers. The hatches are colored in
        blue, orange, and green, respectively. After the first three markers, the colors
        are cycled through again.

    np.random.seed(19680801)

    fig, ax = plt.subplots()

    x = [29, 36, 41, 25, 32, 70, 62, 58, 66, 80, 58, 68, 62, 37, 48]
    y = [82, 76, 48, 53, 62, 70, 84, 68, 55, 75, 29, 25, 12, 17, 20]
    colors = ['tab:blue'] * 5 + ['tab:orange'] * 5 + ['tab:green'] * 5

    ax.scatter(
        x,
        y,
        s=800,
        hatch="xxxx",
        hatchcolor=colors,
        facecolor="none",
        edgecolor="black",
    )

Axis and Ticks
==============

Standard getters/setters for axis inversion state
-------------------------------------------------

Whether an axis is inverted can now be queried using the `.axes.Axes` getters
`~.Axes.get_xinverted`/`~.Axes.get_yinverted` and set using
`~.Axes.set_xinverted`/`~.Axes.set_yinverted`.

The previously existing methods (`.Axes.xaxis_inverted`, `.Axes.invert_xaxis`) are now
discouraged (but not deprecated) due to their non-standard naming and behavior.

``xtick`` and ``ytick`` rotation modes
--------------------------------------

A new feature has been added for handling rotation of xtick and ytick labels more
intuitively. The new `rotation modes <matplotlib.text.Text.set_rotation_mode>` "xtick"
and "ytick" automatically adjust the alignment of rotated tick labels, so that the text
points towards their anchor point, i.e. ticks. This works for all four sides of the plot
(bottom, top, left, right), reducing the need for manual adjustments when rotating
labels.

.. plot::
    :include-source:
    :alt: Example of rotated xtick and ytick labels.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), layout='constrained')

    pos = range(5)
    labels = ['label'] * 5
    ax1.set_xticks(pos, labels, rotation=-45, rotation_mode='xtick')
    ax1.set_yticks(pos, labels, rotation=45, rotation_mode='ytick')
    ax2.xaxis.tick_top()
    ax2.set_xticks(pos, labels, rotation=-45, rotation_mode='xtick')
    ax2.yaxis.tick_right()
    ax2.set_yticks(pos, labels, rotation=45, rotation_mode='ytick')

Improved selection of log-scale ticks
-------------------------------------

The algorithm for selecting log-scale ticks (on powers of ten) has been improved. In
particular, it will now always draw as many ticks as possible (e.g., it will not draw a
single tick if it was possible to fit two ticks); if subsampling ticks, it will prefer
putting ticks on integer multiples of the subsampling stride (e.g., it prefers putting
ticks at 10\ :sup:`0`, 10\ :sup:`3`, 10\ :sup:`6` rather than 10\ :sup:`1`, 10\
:sup:`4`, 10\ :sup:`7`) if this results in the same number of ticks at the end; and it
is now more robust against floating-point calculation errors.

Colors and colormaps
====================

Okabe-Ito accessible color sequence
-----------------------------------

Matplotlib now includes the `Okabe-Ito color sequence
<https://jfly.uni-koeln.de/color/#pallet>`_. Its colors remain distinguishable for
common forms of color-vision deficiency and when printed.

For example, to set it as the default colormap for your plots and image-like artists,
use:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams['axes.prop_cycle'] = cycler(color='okabe_ito')
    plt.rcParams['image.cmap'] = 'okabe_ito'

Or, when creating plots, you can pass it explicitly:

.. plot::
    :alt:
        A plot of 8 lines, with colors of the Okabe-Ito sequence, from bottom to top:
        black, orange, sky blue, bluish green, yellow, blue, vermilion, and reddish
        purple.

    colors = plt.colormaps['okabe_ito'].colors
    x = range(5)
    for i, c in enumerate(colors):
        plt.plot(x, [v*(i+1) for v in x], color=c, label=f'line {i}')
    plt.legend()

Six and eight color Petroff color cycles
----------------------------------------

The six and eight color accessible Petroff color cycles are named ``'petroff6'`` and
``'petroff8'``. They complement the existing ``'petroff10'`` color cycle, added in
:ref:`Matplotlib 3.10.0 <whats-new-3-10-0-petroff10>`.

For more details see `Petroff, M. A.: "Accessible Color Sequences for Data
Visualization" <https://arxiv.org/abs/2107.02270>`_. To load the ``'petroff6'`` color
cycle in place of the default::

  import matplotlib.pyplot as plt
  plt.style.use('petroff6')

.. plot::
    :alt:
        A plot of 6 lines, with colors of the Petroff six-color cycle, from bottom to
        top: blue, orange, red, dark purple, grey, light purple.

    plt.style.use('petroff6')
    x = range(5)
    for i in range(6):
        plt.plot(x, [v*(i+1) for v in x], label=f'line {i}')
    plt.legend()

or to load the ``'petroff8'`` color cycle::

  import matplotlib.pyplot as plt
  plt.style.use('petroff8')

.. plot::
    :alt:
        A plot of 8 lines, with colors of the Petroff eight-color cycle, from bottom to
        top: blue, orange, red, purple, grey, light blue, blue, dark grey.

    plt.style.use('petroff8')
    x = range(5)
    for i in range(8):
        plt.plot(x, [v*(i+1) for v in x], label=f'line {i}')
    plt.legend()

Setting the default color cycle to a named color sequence
---------------------------------------------------------

The default color cycle may now be configured in the ``matplotlibrc`` file or a style
file to use any of the :doc:`/gallery/color/color_sequences`. For example

.. code-block:: none

   axes.prop_cycle : cycler(color='Accent')

Colormaps support giving colors for bad, under and over values on creation
--------------------------------------------------------------------------

Colormaps gained keyword arguments ``bad``, ``under``, and ``over`` to specify these
values on creation. Previously, these values would have to be set afterwards using one
of `~.Colormap.set_bad`, `~.Colormap.set_under`, `~.Colormap.set_bad`,
`~.Colormap.set_extremes`, `~.Colormap.with_extremes`.

It is recommended to use the new functionality, e.g.::

    cmap = ListedColormap(colors, bad="red", under="darkblue", over="purple")

instead of::

    cmap = ListedColormap(colors).with_extremes(
        bad="red", under="darkblue", over="purple")

or::

   cmap = ListedColormap(colors)
   cmap.set_bad("red")
   cmap.set_under("darkblue")
   cmap.set_over("purple")

Tuning transparency of colormaps
--------------------------------

The new method `.Colormap.with_alpha` allows to create a new colormap with the same
color values but a new uniform alpha value. This is handy if you want to modify only the
transparency of mapped colors for an Artist.

Fonts and Text
==============

.. dropdown:: Important
    :color: warning
    :icon: alert

    This release of Matplotlib includes a large update to text processing and rendering.
    Within our published wheels, the bundled version of FreeType has been updated, and
    the rendering pipeline itself has been overhauled to support modern features and
    fonts, as outlined below. It is unfortunately not possible to reproduce the exact
    same pixel values for a piece of rendered text as in previous versions.

    If you are reliant on pixel-perfect consistency between versions, this will be
    broken in this release. For downstream packages that are testing plots, we recommend
    a few options:

    1. Update your test images directly; if you are comfortable with requiring 3.11 and
       above only, then this is the simplest option. However, it does mean dropping
       support for many users with older Matplotlib versions. Alternatively, provide two
       sets of test images, one before and one after. This would increase compatibility
       at the cost of disk space.
    2. Increase tolerances on tests with text. Note that this might obscure unintended
       differences, so be careful with increasing tolerances too high. If you are using
       Matplotlib's `.image_comparison` decorator, you can pass the *tol* argument::

           @image_comparison(['plot.png'], tol=5, style='mpl20')
           def test_plot():
               ...

       If you are using `pytest-mpl`_, then you can pass the *tolerance* argument::

           @pytest.mark.mpl_image_compare(tolerance=5)
           def test_plot():
               ...
    3. Remove non-essential text elements. The easiest way to avoid this problem is to
       not have any text to worry about. For both the `.image_comparison` decorator and
       `pytest-mpl`_, you can pass the *remove_text* argument::

           @image_comparison(['plot.png'], remove_text=True, style='mpl20')
           def test_plot():
               ...

           @pytest.mark.mpl_image_compare(remove_text=True)
           def test_plot():
               ...

       to remove the title and tick texts; note other deliberate texts are not removed.
    4. Replace text with placeholders. If you do need text to exist, but don't intend to
       check the exact pixels (for example, to confirm layout is correct), then you can
       replace the text with a placeholder of a fixed size. If you are using `pytest`_,
       the `.text_placeholders` fixture may be used to replace text with a fixed-size
       box. Consider vendoring a similar fixture in your own tests if necessary.
    5. Use a different image comparison algorithm. While not available in Matplotlib's
       testing framework, a `perceptual hashing algorithm`_ may be more appropriate if
       you wish to avoid depending on exact pixel values.

.. _perceptual hashing algorithm: https://en.wikipedia.org/wiki/Perceptual_hashing
.. _pytest: http://doc.pytest.org/en/latest/
.. _pytest-mpl: https://pytest-mpl.readthedocs.io/

Complex text layout with libraqm
--------------------------------

Text support has been extended to include complex text layout. This support includes:

1. Languages that require advanced layout, such as Arabic or Hebrew.
2. Text that mixes left-to-right and right-to-left languages.

   .. plot::
       :show-source-link: False
       :alt: The mixed-language text 'Here is some رَقْم in اَلْعَرَبِيَّةُ'.

       text = 'Here is some رَقْم in اَلْعَرَبِيَّةُ'
       fig = plt.figure(figsize=(6, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center')

3. Ligatures that combine several adjacent characters for improved legibility.

   .. plot::
       :show-source-link: False
       :alt:
           A rightwards arrow pointing from the individual letters 'f', 'f', and 'i', to
           the 'ffi' ligature.

       text = 'f\N{Hair Space}f\N{Hair Space}i \N{Rightwards Arrow} ffi'
       fig = plt.figure(figsize=(3, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center')

4. Combining multiple or double-width diacritics.

   .. plot::
       :show-source-link: False
       :alt:
           An "equation" showing the letter 'a' plus a circumflex accent plus a tilde
           plus the letter 'c' plus a diaeresis, producing a single glyph of all of them.

       text = (
           'a\N{Combining Circumflex Accent}\N{Combining Double Tilde}'
           'c\N{Combining Diaeresis}')
       text = ' + '.join(
           c if c in 'ac' else f'\N{Dotted Circle}{c}'
           for c in text) + f' \N{Rightwards Arrow} {text}'
       fig = plt.figure(figsize=(6, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center',
                # Builtin DejaVu Sans doesn't support multiple diacritics.
                family=['Noto Sans', 'DejaVu Sans'])

Note, all advanced features require corresponding font support, and may require
additional fonts over the builtin DejaVu Sans.

Specifying font feature tags
----------------------------

OpenType fonts may support feature tags that specify alternate glyph shapes or
substitutions to be made optionally. The text API now supports setting a list of feature
tags to be used with the associated font. Feature tags can be set/get with:

- `matplotlib.text.Text.set_fontfeatures` / `matplotlib.text.Text.get_fontfeatures`
- Any API that creates a `.Text` object by passing the *fontfeatures* argument (e.g.,
  ``plt.xlabel(..., fontfeatures=...)``)

Font feature strings are eventually passed to HarfBuzz, and so all `string formats
supported by hb_feature_from_string()
<https://harfbuzz.github.io/harfbuzz-hb-common.html#hb-feature-from-string>`__ are
supported. Note though that subranges are not explicitly supported and behaviour may
change in the future.

For example, the default font ``DejaVu Sans`` enables Standard Ligatures (the ``'liga'``
tag) by default, and also provides optional Discretionary Ligatures (the ``dlig`` tag.)
These may be toggled with ``+`` or ``-``.

.. plot::
    :include-source:
    :alt:
        An example of ligatures affecting text, in four lines. The first line is the
        title "Ligatures". Each subsequent line shows the style, followed by the
        examples "fi", "ffi", "fl", and "st". The second line is the default, where all
        but "st" use ligatures. The third line has disabled ligatures and all examples
        are drawn as individual glyphs. The fourth line has enabled discretionary
        ligatures and all examples, including the "st", use ligatures.

    fig = plt.figure(figsize=(7, 3))

    fig.text(0.5, 0.85, 'Ligatures', fontsize=40, horizontalalignment='center')

    # Default has Standard Ligatures (liga).
    fig.text(0, 0.6, 'Default: fi ffi fl st', fontsize=40)

    # Disable Standard Ligatures with -liga.
    fig.text(0, 0.35, 'Disabled: fi ffi fl st', fontsize=40,
             fontfeatures=['-liga'])

    # Enable Discretionary Ligatures with dlig.
    fig.text(0, 0.1, 'Discretionary: fi ffi fl st', fontsize=40,
             fontfeatures=['dlig'])

Available font feature tags may be found at
https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist

Specifying text language
------------------------

OpenType fonts may support language systems which can be used to select different
typographic conventions, e.g., localized variants of letters that share a single Unicode
code point, or different default font features. The text API now supports setting a
language to be used and may be set/get with:

- `matplotlib.text.Text.set_language` / `matplotlib.text.Text.get_language`
- Any API that creates a `.Text` object by passing the *language* argument (e.g.,
  ``plt.xlabel(..., language=...)``)

The language of the text must be in a format accepted by libraqm, namely `a BCP47
language code <https://www.w3.org/International/articles/language-tags/>`_. If None or
unset, then no particular language will be implied, and default font settings will be
used.

For example, Matplotlib's default font ``DejaVu Sans`` supports language-specific glyphs
in the Serbian and Macedonian languages in the Cyrillic alphabet (vs Russian), or the
Sámi family of languages in the Latin alphabet (vs English).

.. plot::
    :include-source:
    :alt:
        An example of how text language affects rendering, in four lines. The first line
        lists the Unicode code point '\U00000431`, which is the Cyrillic small letter BE.
        The second line then shows the rendering when the language is set to Serbian vs
        set to Russian. The third line lists the Unicode code point '\U0000014a`, which
        is the Latin capital letter ENG. The fourth line then shows the rendering when
        the language is set to Inari Sámi vs set to English.

    fig = plt.figure(figsize=(7, 3))

    char = '\U00000431'
    fig.text(0.5, 0.8, f'\\U{ord(char):08x}', fontsize=40, horizontalalignment='center')
    fig.text(0, 0.6, f'Serbian: {char}', fontsize=40, language='sr')
    fig.text(1, 0.6, f'Russian: {char}', fontsize=40, language='ru',
             horizontalalignment='right')

    char = '\U0000014a'
    fig.text(0.5, 0.3, f'\\U{ord(char):08x}', fontsize=40, horizontalalignment='center')
    fig.text(0, 0.1, f'Inari Sámi: {char}', fontsize=40, language='smn')
    fig.text(1, 0.1, f'English: {char}', fontsize=40, language='en',
             horizontalalignment='right')

Missing glyphs use Last Resort font
-----------------------------------

Most fonts do not have 100% character coverage, and will fall back to a "not found"
glyph for characters that are not provided. Often, this glyph will be minimal (e.g., the
default DejaVu Sans "not found" glyph is just a rectangle.) Such minimal glyphs provide
no context as to the characters that are missing.

Now, missing glyphs will fall back to the `Last Resort font
<https://github.com/unicode-org/last-resort-font>`__ produced by the Unicode Consortium.
This special-purpose font provides glyphs that represent types of Unicode characters.
These glyphs show a representative character from the missing Unicode block, and at
larger sizes, more context to help determine which character and font are needed.

To disable this fallback behaviour, set :rc:`font.enable_last_resort` to ``False``.

.. plot::
    :alt:
        An example of missing glyph behaviour, the first glyph from Bengali script,
        second glyph from Hiragana, and the last glyph from the Unicode Private Use
        Area. Multiple lines repeat the text with increasing font size from top to
        bottom.

    text_raw = r"'\N{Bengali Digit Zero}\N{Hiragana Letter A}\ufdd0'"
    text = eval(text_raw)
    sizes = [
        (0.85, 8),
        (0.80, 10),
        (0.75, 12),
        (0.70, 16),
        (0.63, 20),
        (0.55, 24),
        (0.45, 32),
        (0.30, 48),
        (0.10, 64),
    ]

    fig = plt.figure()
    fig.text(0.01, 0.90, f'Input: {text_raw}')
    for y, size in sizes:
        fig.text(0.01, y, f'{size}pt:{text}', fontsize=size)

Fonts addressable by all their SFNT family names
------------------------------------------------

Fonts can now be selected by any of the family names they advertise in the OpenType name
table, not just the one FreeType reports as the primary family name.

Some fonts store different family names on different platforms or in different
name-table entries.  For example, Ubuntu Light stores ``"Ubuntu"`` in the
Macintosh-platform Name ID 1 slot (which FreeType uses as the primary name) and
``"Ubuntu Light"`` in the Microsoft-platform Name ID 1 slot.  Previously only the
FreeType-derived name was registered, requiring an obscure weight-based workaround::

    # Previously required
    matplotlib.rcParams['font.family'] = 'Ubuntu'
    matplotlib.rcParams['font.weight'] = 300

All name-table entries that describe a family — Name ID 1 on both platforms, the
Typographic Family (Name ID 16), and the WWS Family (Name ID 21) — are now registered as
separate entries in the `~matplotlib.font_manager.FontManager`, so any of those names
can be used directly::

    matplotlib.rcParams['font.family'] = 'Ubuntu Light'

Support for loading TrueType Collection fonts
---------------------------------------------

TrueType Collection fonts (commonly found as files with a ``.ttc`` extension) are now
supported. Namely, Matplotlib will include these file extensions in its scan for system
fonts, and will add all sub-fonts to its list of available fonts (i.e., the list from
`~.font_manager.get_font_names`).

From most high-level API, this means you should be able to specify the name of any
sub-font in a collection just as you would any other font. Note that at this time, there
is no way to specify the entire collection with any sort of automated selection of the
internal sub-fonts.

In the low-level API, to ensure backwards-compatibility while facilitating this new
support, a `.FontPath` instance (comprised of a font path and a sub-font index, with
behaviour similar to a `str`) may be passed to the font management API in place of a
simple `os.PathLike` path. Any font management API that previously returned a string
path now returns a `.FontPath` instance instead.

New environment variable to ignore system fonts
-----------------------------------------------

System fonts may be ignored by setting the :envvar:`MPL_IGNORE_SYSTEM_FONTS`; this
suppresses searching for system fonts (in known directories or via some
platform-specific subprocess) as well as limiting the results from
`.FontManager.findfont`.

Mathtext distinguishes *italic* and *normal* font
-------------------------------------------------

Matplotlib's lightweight TeX expression parser (``usetex=False``) now distinguishes
between *italic* and *normal* math fonts to closer replicate the behaviour of LaTeX. The
normal math font is selected by default in math environment (unless the rcParam
``mathtext.default`` is overwritten) but can be explicitly set with the new
``\mathnormal`` command. Italic font is selected with ``\mathit``. The main difference
is that *italic* produces italic digits, whereas *normal* produces upright digits.
Previously, it was not possible to typeset italic digits. Note that ``normal`` now
corresponds to what used to be ``it``, whereas ``it`` now renders all characters italic.
**Important**: In case the default mathematics font is overwritten by setting
``mathtext.default: it`` in ``matplotlibrc``, it must be either commented out or changed
to ``mathtext.default: normal`` to preserve its behaviour. Otherwise, all alphanumeric
characters, including digits, are rendered italic.

One difference to traditional LaTeX is that LaTeX further distinguishes between *normal*
(``\mathnormal``) and *default math*, where the default uses roman digits and normal
uses oldstyle digits. This distinction is no longer present with modern LaTeX engines
and unicode-math nor in Matplotlib.

mathtext support for ``\phantom``, ``\llap``, ``\rlap``
-------------------------------------------------------

mathtext gained support for the TeX macros ``\phantom``, ``\llap``, and ``\rlap``.
``\phantom`` allows to occupy some space on the canvas as if some text was being
rendered, without actually rendering that text, whereas ``\llap`` and ``\rlap`` allows
to render some text on the canvas while pretending that it occupies no space.
Altogether these macros allow some finer control of text alignments.

See https://www.tug.org/TUGboat/tb22-4/tb72perlS.pdf for a detailed description of these
macros.

For example, using these macros in the first legend below allows reserving space so that
it is the same size as the second legend with longer text:

.. plot::
    :include-source:
    :alt:
        Two plots of diagonal lines. The first Axes has a legend with a single entry
        labelled "foo". The second Axes has a legend with two entries labelled "foo" and
        "a longer label". Both legends are the same width despite the former containing
        a shorter label.

    fig = plt.figure(layout="constrained")
    sfs = fig.subfigures(2)

    ax0 = sfs[0].add_subplot()
    ax0.plot([1, 2], label=r"$\rlap{\text{foo}}\phantom{\text{a longer label}}$")
    sfs[0].legend(loc="outside right upper")

    ax1 = sfs[1].add_subplot()
    ax1.plot([1, 2], label="foo")
    ax1.plot([2, 1], label="a longer label")
    sfs[1].legend(loc="outside right upper")

Underlining text while using Mathtext
-------------------------------------

Mathtext now supports the ``\underline`` command.

.. plot::
    :include-source:
    :alt:
        Two lines of text. The first says "This is underlined text." and the word
        "underlined" is italic and has a line under it. The second line says "So us
        this." and the word "this" has a line under it.

    plt.figure(figsize=(6, 2))
    plt.text(0.05, 0.7, r'This is $\underline{underlined}$ text.', fontsize=24)
    plt.text(0.05, 0.2, r'So is $\underline{\mathrm{this}}$.', fontsize=24)
    plt.axis('off')

Improved font embedding in PDF
------------------------------

Both Type 3 and Type 42 fonts (see :ref:`fonts` for more details) are now embedded into
PDFs without limitation. Fonts may be split into multiple embedded subsets in order to
satisfy format limits. Additionally, a corrected Unicode mapping is added for each.

This means that *all* text should now be selectable and copyable in PDF viewers that
support doing so.

When using the ``usetex`` feature, Matplotlib calls TeX to render the text and formulas
in the figure. The fonts that get used are usually "Type 1" fonts. They used to be
embedded in full but are now limited to the glyphs that are actually used in the figure.
This reduces the size of the resulting PDF files.

rcParams improvements
=====================

Separate styling options for major/minor grid line in rcParams
--------------------------------------------------------------

Using :rc:`grid.major.*` or :rc:`grid.minor.*` will overwrite the value in :rc:`grid.*`
for the major and minor gridlines, respectively.

.. plot::
    :include-source:
    :alt: Modifying the gridlines using the new options `rcParams`

    import matplotlib as mpl

    # Set visibility for major and minor gridlines
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["axes.grid.which"] = "both"

    # Using grid.* to set both major and minor properties
    mpl.rcParams["grid.color"] = "lightgrey"

    # Overwrite some values for major and minor separately
    mpl.rcParams["grid.major.linewidth"] = 1.2
    mpl.rcParams["grid.minor.color"] = "tab:blue"
    mpl.rcParams["grid.minor.linestyle"] = ":"

    plt.plot([0, 1], [0, 1])

``axes.prop_cycle`` rcParam security improvements
-------------------------------------------------

The ``axes.prop_cycle`` rcParam is now parsed in a safer and more restricted manner.
Only literals, ``cycler()`` and ``concat()`` calls, the operators ``+`` and ``*``, and
slicing are allowed. All previously valid cycler strings documented at
https://matplotlib.org/cycler/ are still supported, for example:

.. code-block:: none

   axes.prop_cycle : cycler('color', ['r', 'g', 'b']) + cycler('linewidth', [1, 2, 3])
   axes.prop_cycle : 2 * cycler('color', 'rgb')
   axes.prop_cycle : concat(cycler('color', 'rgb'), cycler('color', 'cmk'))
   axes.prop_cycle : cycler('color', 'rgbcmk')[:3]

Legends
=======

``legend.linewidth`` rcParam and parameter
------------------------------------------

A new rcParam ``legend.linewidth`` has been added to control the line width of the
legend's box edges. When set to ``None`` (the default), it inherits the value from
``patch.linewidth``. This allows for independent control of the legend frame line width
without affecting other elements.

The `.Legend` constructor also accepts a new *linewidth* parameter to set the legend
frame line width directly, overriding the rcParam value.

.. plot::
    :include-source:
    :alt: A line plot with a legend showing a thick border around the legend box.

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], label='data')
    ax.legend(linewidth=2.0)  # Thick legend box edge

``PatchCollection`` legends now supported
-----------------------------------------

`.PatchCollection` instances now properly display in legends when given a label.
Previously, labels on `~.PatchCollection` objects were ignored by the legend system,
requiring users to create manual legend entries.

.. plot::
   :include-source:
   :alt:
       The legend entry displays a rectangle matching the visual properties (colors,
       line styles, line widths) of the first patch in the collection.

    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots()
    patches = [mpatches.Circle((0, 0.5), 0.1), mpatches.Rectangle((0.5, 0), 0.2, 0.3)]
    pc = PatchCollection(patches, facecolor='blue', edgecolor='black', label='My patches')
    ax.add_collection(pc)
    ax.legend()  # Now displays the label "My patches"

Widgets and Interactivity
=========================

Zooming using mouse wheel
-------------------------

:kbd:`Control+MouseWheel` can be used to zoom in the plot windows. Additionally,
:kbd:`x+MouseWheel` zooms only the x-axis and :kbd:`y+MouseWheel` zooms only the y-axis.

The zoom focusses on the mouse pointer. With :kbd:`Control`, the axes aspect ratio is
kept; with :kbd:`x` or :kbd:`y`, only the respective axis is scaled.

Zooming is currently only supported on rectilinear Axes.

Consistent zoom boxes
---------------------

Zooming now has a consistent dashed box style across all backends.

RadioButtons and CheckButtons widgets support flexible layouts
--------------------------------------------------------------

The `.widgets.RadioButtons` and `.widgets.CheckButtons` widgets now support arranging
buttons in different layouts via the new *layout* parameter. You can arrange buttons
vertically (default), horizontally, or in a 2D grid by passing a ``(rows, cols)`` tuple.

See :doc:`/gallery/widgets/radio_buttons_grid` for a ``(rows, cols)`` example.

.. plot::
    :include-source:
    :alt: Multiple sine waves with checkboxes to toggle their visibility.

    from matplotlib.widgets import CheckButtons

    t = np.arange(0.0, 2.0, 0.01)
    s0 = np.sin(2*np.pi*t)
    s1 = np.sin(4*np.pi*t)
    s2 = np.sin(6*np.pi*t)
    s3 = np.sin(8*np.pi*t)

    fig, axes = plt.subplot_mosaic(
        [['main'], ['buttons']],
        height_ratios=[8, 1],
        layout="constrained",
    )

    l0, = axes['main'].plot(t, s0, lw=2, label='2 Hz')
    l1, = axes['main'].plot(t, s1, lw=2, label='4 Hz')
    l2, = axes['main'].plot(t, s2, lw=2, label='6 Hz')
    l3, = axes['main'].plot(t, s3, lw=2, label='8 Hz')
    axes['main'].set_xlabel('Time (s)')
    axes['main'].set_ylabel('Amplitude')

    lines_by_label = {l.get_label(): l for l in [l0, l1, l2, l3]}

    axes['buttons'].set_facecolor('0.95')
    check = CheckButtons(
        axes['buttons'],
        labels=lines_by_label.keys(),
        actives=[l.get_visible() for l in lines_by_label.values()],
        layout='horizontal'
    )

    def callback(label):
        ln = lines_by_label[label]
        ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(callback)

Callable *valfmt* for ``Slider`` and ``RangeSlider``
----------------------------------------------------

In addition to the existing %-format string, the *valfmt* parameter of
`~.matplotlib.widgets.Slider` and `~.matplotlib.widgets.RangeSlider` now also accepts a
callable of the form ``valfmt(val: float) -> str``.

WebAgg scroll capture control
------------------------------

The WebAgg backend now provides the ability to capture scroll events to prevent page
scrolling when interacting with plots. This can be enabled or disabled via the new
`.FigureCanvasWebAggCore.set_capture_scroll` and
`.FigureCanvasWebAggCore.get_capture_scroll` methods.

3D plotting improvements
========================

Non-linear scales on 3D axes
----------------------------

Resolving a long-standing issue, 3D axes now support non-linear axis scales such as
``'log'``, ``'symlog'``, ``'logit'``, ``'asinh'``, and custom ``'function'`` scales,
just like 2D axes. Use `~.Axes3D.set_xscale`, `~.Axes3D.set_yscale`, and
`~.Axes3D.set_zscale` to set the scale for each axis independently.

.. plot::
    :include-source:
    :alt: A 3D plot with a linear x-axis, logarithmic y-axis, and symlog z-axis.

    # A sine chirp with increasing frequency and amplitude
    x = np.linspace(0, 1, 400)  # time
    y = 10 ** (2 * x)  # frequency, growing exponentially from 1 to 100 Hz
    phase = 2 * np.pi * (10 ** (2 * x) - 1) / (2 * np.log(10))
    z = np.sin(phase) * x ** 2 * 10  # amplitude, growing quadratically

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('Time (linear)')
    ax.set_ylabel('Frequency, Hz (log)')
    ax.set_zlabel('Amplitude (symlog)')

    ax.set_yscale('log')
    ax.set_zscale('symlog')

See `matplotlib.scale` for details on all available scales and their parameters.

Snapping 3D rotation angles with Control key
--------------------------------------------

Rotation of 3D axes now supports snapping to fixed angular increments when holding the
:kbd:`Control` key during mouse rotation.

The snap step size is controlled by the new :rc:`axes3d.snap_rotation` rcParam. Setting
it to 0 disables snapping.

For example::

    mpl.rcParams["axes3d.snap_rotation"] = 10

will snap elevation, azimuth, and roll angles to multiples of 10 degrees while rotating
with the mouse.

3D depth-shading fix
--------------------

Previously, a slightly buggy method of estimating the visual "depth" of 3D items could
lead to sudden and unexpected changes in transparency as the plot orientation changed.

Now, the behavior has been made smooth and predictable. A new parameter
*depthshade_minalpha* has also been added to allow users to set the minimum transparency
level. Depth-shading is an option for `.Patch3DCollection` and `.Path3DCollection`,
including 3D scatter plots.

The default values for ``depthshade`` and ``depthshade_minalpha`` are now controlled by
:rc:`axes3d.depthshade` and :rc:`axes3d.depthshade_minalpha`, respectively.

A simple example:

.. plot::
    :include-source:
    :alt: A 3D scatter plot with depth-shading enabled.

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    X = [i for i in range(10)]
    Y = [i for i in range(10)]
    Z = [i for i in range(10)]
    S = [(i + 1) * 400 for i in range(10)]

    ax.scatter(
        xs=X, ys=Y, zs=Z, s=S,
        depthshade=True,
        depthshade_minalpha=0.3,
    )
    ax.view_init(elev=10, azim=-150, roll=0)

3D performance improvements
---------------------------

Draw time for 3D plots has been improved, especially for surface and wireframe plots.
Users should see up to a 10× speedup in some cases. This should make interacting with 3D
plots much more responsive.

Other improvements
==================

Saving figures as GIF works again
---------------------------------

According to the figure documentation, the ``savefig`` method supports the GIF format
with the file extension ``.gif``. However, GIF support had been broken since Matplotlib
2.0.0. It works again.

``CallbackRegistry.disconnect`` allows directly callbacks by function
-------------------------------------------------------------------------

`.CallbackRegistry` now allows directly passing a function and optionally signal to
`~.CallbackRegistry.disconnect` instead of needing to track the callback ID returned by
`~.CallbackRegistry.connect`.

.. code-block:: python

    from matplotlib.cbook import CallbackRegistry

    def my_callback(event):
        print(event)

    callbacks = CallbackRegistry()
    callbacks.connect('my_signal', my_callback)

    # Disconnect by function reference instead of callback ID
    callbacks.disconnect('my_signal', my_callback)

``violin_stats`` simpler *method* parameter
-------------------------------------------

The *method* parameter of `~.cbook.violin_stats` may now be specified as tuple of
strings, and has a new default ``("GaussianKDE", "scott")``.  Calling
`~.cbook.violin_stats` followed by `~.Axes.violin` is therefore now equivalent to
calling `~.Axes.violinplot`.

.. plot::
    :include-source:
    :alt:
        Example showing violin_stats followed by violin gives the same result as
        violinplot.

    from matplotlib.cbook import violin_stats

    rng = np.random.default_rng(19680801)
    data = rng.normal(size=(10, 3))

    fig, (ax1, ax2) = plt.subplots(ncols=2, layout='constrained', figsize=(6.4, 3.5))

    # Create the violin plot in one step
    ax1.violinplot(data)
    ax1.set_title('One Step')

    # Process the data and then create the violin plot
    vstats = violin_stats(data)
    ax2.violin(vstats)
    ax2.set_title('Two Steps')
