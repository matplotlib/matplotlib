=============================================
What's new in Matplotlib 3.9.0 (May 15, 2024)
=============================================

For a list of all of the issues and pull requests since the last revision, see the
:ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 4

.. toctree::
   :maxdepth: 4

Plotting and Annotation improvements
====================================

``Axes.inset_axes`` is no longer experimental
---------------------------------------------

`.Axes.inset_axes` is considered stable for use.

Legend support for Boxplot
--------------------------

Boxplots now support a *label* parameter to create legend entries. Legend labels can be
passed as a list of strings to label multiple boxes in a single `.Axes.boxplot` call:

.. plot::
    :include-source:
    :alt: Example of creating 3 boxplots and assigning legend labels as a sequence.

    np.random.seed(19680801)
    fruit_weights = [
        np.random.normal(130, 10, size=100),
        np.random.normal(125, 20, size=100),
        np.random.normal(120, 30, size=100),
    ]
    labels = ['peaches', 'oranges', 'tomatoes']
    colors = ['peachpuff', 'orange', 'tomato']

    fig, ax = plt.subplots()
    ax.set_ylabel('fruit weight (g)')

    bplot = ax.boxplot(fruit_weights,
                       patch_artist=True,  # fill with color
                       label=labels)

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks([])
    ax.legend()


Or as a single string to each individual `.Axes.boxplot`:

.. plot::
    :include-source:
    :alt: Example of creating 2 boxplots and assigning each legend label as a string.

    fig, ax = plt.subplots()

    data_A = np.random.random((100, 3))
    data_B = np.random.random((100, 3)) + 0.2
    pos = np.arange(3)

    ax.boxplot(data_A, positions=pos - 0.2, patch_artist=True, label='Box A',
               boxprops={'facecolor': 'steelblue'})
    ax.boxplot(data_B, positions=pos + 0.2, patch_artist=True, label='Box B',
               boxprops={'facecolor': 'lightblue'})

    ax.legend()

Percent sign in pie labels auto-escaped with ``usetex=True``
------------------------------------------------------------

It is common, with `.Axes.pie`, to specify labels that include a percent sign (``%``),
which denotes a comment for LaTeX. When enabling LaTeX with :rc:`text.usetex` or passing
``textprops={"usetex": True}``, this used to cause the percent sign to disappear.

Now, the percent sign is automatically escaped (by adding a preceding backslash) so that
it appears regardless of the ``usetex`` setting. If you have pre-escaped the percent
sign, this will be detected, and remain as is.

``hatch`` parameter for stackplot
---------------------------------

The `~.Axes.stackplot` *hatch* parameter now accepts a list of strings describing
hatching styles that will be applied sequentially to the layers in the stack:

.. plot::
    :include-source:
    :alt: Two charts, identified as ax1 and ax2, showing "stackplots", i.e. one-dimensional distributions of data stacked on top of one another. The first plot, ax1 has cross-hatching on all slices, having been given a single string as the "hatch" argument. The second plot, ax2 has different styles of hatching on each slice - diagonal hatching in opposite directions on the first two slices, cross-hatching on the third slice, and open circles on the fourth.

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))

    cols = 10
    rows = 4
    data = (
    np.reshape(np.arange(0, cols, 1), (1, -1)) ** 2
    + np.reshape(np.arange(0, rows), (-1, 1))
    + np.random.random((rows, cols))*5
    )
    x = range(data.shape[1])
    ax1.stackplot(x, data, hatch="x")
    ax2.stackplot(x, data, hatch=["//","\\","x","o"])

    ax1.set_title("hatch='x'")
    ax2.set_title("hatch=['//','\\\\','x','o']")

    plt.show()

Add option to plot only one half of violin plot
-----------------------------------------------

Setting the parameter *side* to 'low' or 'high' allows to only plot one half of the
`.Axes.violinplot`.

.. plot::
    :include-source:
    :alt: Three copies of a vertical violin plot; first in blue showing the default of both sides, followed by an orange copy that only shows the "low" (or left, in this case) side, and finally a green copy that only shows the "high" (or right) side.

    # Fake data with reproducible random state.
    np.random.seed(19680801)
    data = np.random.normal(0, 8, size=100)

    fig, ax = plt.subplots()

    ax.violinplot(data, [0], showmeans=True, showextrema=True)
    ax.violinplot(data, [1], showmeans=True, showextrema=True, side='low')
    ax.violinplot(data, [2], showmeans=True, showextrema=True, side='high')

    ax.set_title('Violin Sides Example')
    ax.set_xticks([0, 1, 2], ['Default', 'side="low"', 'side="high"'])
    ax.set_yticklabels([])

``axhline`` and ``axhspan`` on polar axes
-----------------------------------------

... now draw circles and circular arcs (`~.Axes.axhline`) or annuli and wedges
(`~.Axes.axhspan`).

.. plot::
    :include-source:
    :alt: A sample polar plot, that contains an axhline at radius 1, an axhspan annulus between radius 0.8 and 0.9, and an axhspan wedge between radius 0.6 and 0.7 and 288° and 324°.

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_rlim(0, 1.2)

    ax.axhline(1, c="C0", alpha=.5)
    ax.axhspan(.8, .9, fc="C1", alpha=.5)
    ax.axhspan(.6, .7, .8, .9, fc="C2", alpha=.5)

Subplot titles can now be automatically aligned
-----------------------------------------------

Subplot axes titles can be misaligned vertically if tick labels or xlabels are placed at
the top of one subplot. The new `~.Figure.align_titles` method on the `.Figure` class
will now align the titles vertically.

.. plot::
    :include-source:
    :alt: A figure with two Axes side-by-side, the second of which with ticks on top. The Axes titles and x-labels appear unaligned with each other due to these ticks.

    fig, axs = plt.subplots(1, 2, layout='constrained')

    axs[0].plot(np.arange(0, 1e6, 1000))
    axs[0].set_title('Title 0')
    axs[0].set_xlabel('XLabel 0')

    axs[1].plot(np.arange(1, 0, -0.1) * 2000, np.arange(1, 0, -0.1))
    axs[1].set_title('Title 1')
    axs[1].set_xlabel('XLabel 1')
    axs[1].xaxis.tick_top()
    axs[1].tick_params(axis='x', rotation=55)

.. plot::
    :include-source:
    :alt: A figure with two Axes side-by-side, the second of which with ticks on top. Unlike the previous figure, the Axes titles and x-labels appear aligned.

    fig, axs = plt.subplots(1, 2, layout='constrained')

    axs[0].plot(np.arange(0, 1e6, 1000))
    axs[0].set_title('Title 0')
    axs[0].set_xlabel('XLabel 0')

    axs[1].plot(np.arange(1, 0, -0.1) * 2000, np.arange(1, 0, -0.1))
    axs[1].set_title('Title 1')
    axs[1].set_xlabel('XLabel 1')
    axs[1].xaxis.tick_top()
    axs[1].tick_params(axis='x', rotation=55)

    fig.align_labels()
    fig.align_titles()

``axisartist`` can now be used together with standard ``Formatters``
--------------------------------------------------------------------

... instead of being limited to axisartist-specific ones.

Toggle minorticks on Axis
-------------------------

Minor ticks on an `~matplotlib.axis.Axis` can be displayed or removed using
`~matplotlib.axis.Axis.minorticks_on` and `~matplotlib.axis.Axis.minorticks_off`; e.g.,
``ax.xaxis.minorticks_on()``. See also `~matplotlib.axes.Axes.minorticks_on`.

``StrMethodFormatter`` now respects ``axes.unicode_minus``
----------------------------------------------------------

When formatting negative values, `.StrMethodFormatter` will now use unicode minus signs
if :rc:`axes.unicode_minus` is set.

    >>> from matplotlib.ticker import StrMethodFormatter
    >>> with plt.rc_context({'axes.unicode_minus': False}):
    ...     formatter = StrMethodFormatter('{x}')
    ...     print(formatter.format_data(-10))
    -10

    >>> with plt.rc_context({'axes.unicode_minus': True}):
    ...     formatter = StrMethodFormatter('{x}')
    ...     print(formatter.format_data(-10))
    −10

Figure, Axes, and Legend Layout
===============================

Subfigures now have controllable zorders
----------------------------------------

Previously, setting the zorder of a subfigure had no effect, and those were plotted on
top of any figure-level artists (i.e for example on top of fig-level legends). Now,
subfigures behave like any other artists, and their zorder can be controlled, with
default a zorder of 0.

.. plot::
    :include-source:
    :alt: Example on controlling the zorder of a subfigure

    x = np.linspace(1, 10, 10)
    y1, y2 = x, -x
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2)
    for subfig in subfigs:
        axarr = subfig.subplots(2, 1)
        for ax in axarr.flatten():
            (l1,) = ax.plot(x, y1, label="line1")
            (l2,) = ax.plot(x, y2, label="line2")
    subfigs[0].set_zorder(6)
    l = fig.legend(handles=[l1, l2], loc="upper center", ncol=2)

Getters for xmargin, ymargin and zmargin
----------------------------------------

`.Axes.get_xmargin`, `.Axes.get_ymargin` and `.Axes3D.get_zmargin` methods have been
added to return the margin values set by `.Axes.set_xmargin`, `.Axes.set_ymargin` and
`.Axes3D.set_zmargin`, respectively.

Mathtext improvements
=====================

``mathtext`` documentation improvements
---------------------------------------

The documentation is updated to take information directly from the parser. This means
that (almost) all supported symbols, operators, etc. are shown at :ref:`mathtext`.

``mathtext`` spacing corrections
--------------------------------

As consequence of the updated documentation, the spacing on a number of relational and
operator symbols were correctly classified and therefore will be spaced properly.

Widget Improvements
===================

Check and Radio Button widgets support clearing
-----------------------------------------------

The `.CheckButtons` and `.RadioButtons` widgets now support clearing their state by
calling their ``.clear`` method. Note that it is not possible to have no selected radio
buttons, so the selected option at construction time is selected.

3D plotting improvements
========================

Setting 3D axis limits now set the limits exactly
-------------------------------------------------

Previously, setting the limits of a 3D axis would always add a small margin to the
limits. Limits are now set exactly by default. The newly introduced rcparam
``axes3d.automargin`` can be used to revert to the old behavior where margin is
automatically added.

.. plot::
    :include-source:
    :alt: Example of the new behavior of 3D axis limits, and how setting the rcParam reverts to the old behavior.

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    plt.rcParams['axes3d.automargin'] = True
    axs[0].set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title='Old Behavior')

    plt.rcParams['axes3d.automargin'] = False  # the default in 3.9.0
    axs[1].set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title='New Behavior')

Other improvements
==================

BackendRegistry
---------------

New :class:`~matplotlib.backends.registry.BackendRegistry` class is the single source of
truth for available backends. The singleton instance is
``matplotlib.backends.backend_registry``. It is used internally by Matplotlib, and also
IPython (and therefore Jupyter) starting with IPython 8.24.0.

There are three sources of backends: built-in (source code is within the Matplotlib
repository), explicit ``module://some.backend`` syntax (backend is obtained by loading
the module), or via an entry point (self-registering backend in an external package).

To obtain a list of all registered backends use:

    >>> from matplotlib.backends import backend_registry
    >>> backend_registry.list_all()

Add ``widths``, ``heights`` and ``angles`` setter to ``EllipseCollection``
--------------------------------------------------------------------------

The ``widths``, ``heights`` and ``angles`` values of the
`~matplotlib.collections.EllipseCollection` can now be changed after the collection has
been created.

.. plot::
    :include-source:

    from matplotlib.collections import EllipseCollection

    rng = np.random.default_rng(0)

    widths = (2, )
    heights = (3, )
    angles = (45, )
    offsets = rng.random((10, 2)) * 10

    fig, ax = plt.subplots()

    ec = EllipseCollection(
        widths=widths,
        heights=heights,
        angles=angles,
        offsets=offsets,
        units='x',
        offset_transform=ax.transData,
        )

    ax.add_collection(ec)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)

    new_widths = rng.random((10, 2)) * 2
    new_heights = rng.random((10, 2)) * 3
    new_angles = rng.random((10, 2)) * 180

    ec.set(widths=new_widths, heights=new_heights, angles=new_angles)

``image.interpolation_stage`` rcParam
-------------------------------------

This new rcParam controls whether image interpolation occurs in "data" space or in
"rgba" space.

Arrow patch position is now modifiable
--------------------------------------

A setter method has been added that allows updating the position of the `.patches.Arrow`
object without requiring a full re-draw.

.. plot::
    :include-source:
    :alt: Example of changing the position of the arrow with the new ``set_data`` method.

    from matplotlib import animation
    from matplotlib.patches import Arrow

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    a = Arrow(2, 0, 0, 10)
    ax.add_patch(a)


    # code for modifying the arrow
    def update(i):
        a.set_data(x=.5, dx=i, dy=6, width=2)


    ani = animation.FuncAnimation(fig, update, frames=15, interval=90, blit=False)

    plt.show()

NonUniformImage now has mouseover support
-----------------------------------------

When mousing over a `~matplotlib.image.NonUniformImage`, the data values are now
displayed.
