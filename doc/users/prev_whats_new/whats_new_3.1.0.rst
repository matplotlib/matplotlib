
What's new in Matplotlib 3.1
============================

For a list of all of the issues and pull requests since the last
revision, see the :ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 4

.. toctree::
   :maxdepth: 4

New Features
------------

`~.dates.ConciseDateFormatter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic date formatter used by default can be quite verbose.  A new
formatter can be accessed that tries to make the tick labels appropriately
concise.

  .. plot::

    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # make a timeseries...
    base = datetime.datetime(2005, 2, 1)
    dates = np.array([base + datetime.timedelta(hours= 2 * i)
                      for i in range(732)])
    N = len(dates)
    np.random.seed(19680801)
    y = np.cumsum(np.random.randn(N))

    lims = [(np.datetime64('2005-02'), np.datetime64('2005-04')),
            (np.datetime64('2005-02-03'), np.datetime64('2005-02-15')),
            (np.datetime64('2005-02-03 11:00'), np.datetime64('2005-02-04 13:20'))]
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    for nn, ax in enumerate(axs):
        # activate the formatter here.
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(dates, y)
        ax.set_xlim(lims[nn])
    axs[0].set_title('Concise Date Formatter')

    plt.show()

Secondary x/y Axis support
~~~~~~~~~~~~~~~~~~~~~~~~~~

A new method provides the ability to add a second axis to an existing
axes via `.Axes.secondary_xaxis` and `.Axes.secondary_yaxis`.  See
:doc:`/gallery/subplots_axes_and_figures/secondary_axis` for examples.

.. plot::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(360))
    ax.secondary_xaxis('top', functions=(np.deg2rad, np.rad2deg))


`~.scale.FuncScale` for arbitrary axes scales
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new `~.scale.FuncScale` class was added (and `~.scale.FuncTransform`)
to allow the user to have arbitrary scale transformations without having to
write a new subclass of `~.scale.ScaleBase`.  This can be accessed by::

  ax.set_yscale('function', functions=(forward, inverse))

where ``forward`` and ``inverse`` are callables that return the scale
transform and its inverse.  See the last example in
:doc:`/gallery/scales/scales`.


Legend for scatter
~~~~~~~~~~~~~~~~~~

A new method for creating legends for scatter plots has been
introduced.  Previously, in order to obtain a legend for a
:meth:`~.axes.Axes.scatter` plot, one could either plot several
scatters, each with an individual label, or create proxy artists to
show in the legend manually.  Now,
:class:`~.collections.PathCollection` provides a method
:meth:`~.collections.PathCollection.legend_elements` to obtain the
handles and labels for a scatter plot in an automated way. This makes
creating a legend for a scatter plot as easy as

.. plot::

    scatter = plt.scatter([1,2,3], [4,5,6], c=[7,2,3])
    plt.legend(*scatter.legend_elements())

An example can be found in :ref:`automatedlegendcreation`.


Matplotlib no longer requires framework app build on MacOSX backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous versions of matplotlib required a Framework build of python to
work. The app type was updated to no longer require this, so the MacOSX
backend should work with non-framework python.


This also adds support for the MacOSX backend for PyPy3.


Figure, FigureCanvas, and Backends
----------------------------------

Figure.frameon is now a direct proxy for the Figure patch visibility state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accessing ``Figure.frameon`` (including via ``get_frameon`` and ``set_frameon``
now directly forwards to the visibility of the underlying Rectangle artist
(``Figure.patch.get_frameon``, ``Figure.patch.set_frameon``).


*pil_kwargs* argument added to savefig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib uses Pillow to handle saving to the JPEG and TIFF formats.  The
`~.Figure.savefig()` function gained a *pil_kwargs* keyword argument, which can
be used to forward arguments to Pillow's `pillow.Image.save()`.

The *pil_kwargs* argument can also be used when saving to PNG.  In that case,
Matplotlib also uses Pillow's `pillow.Image.save()` instead of going through its
own builtin PNG support.


Add ``inaxes`` method to `.FigureCanvasBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.FigureCanvasBase` class has now an `~.FigureCanvasBase.inaxes`
method to check whether a point is in an axes and returns the topmost
axes, else None.

cairo backend defaults to pycairo instead of cairocffi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This leads to faster import/runtime performance in some cases. The backend
will fall back to cairocffi in case pycairo isn't available.


Axes and Artists
----------------

axes_grid1 and axisartist Axes no longer draw spines twice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, spines of `.axes_grid1` and `.axisartist` Axes would be drawn twice,
leading to a "bold" appearance.  This is no longer the case.


Return type of ArtistInspector.get_aliases changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.ArtistInspector.get_aliases` previously returned the set of aliases as
``{fullname: {alias1: None, alias2: None, ...}}``.  The dict-to-None mapping
was used to simulate a set in earlier versions of Python.  It has now been
replaced by a set, i.e. ``{fullname: {alias1, alias2, ...}}``.

This value is also stored in `.ArtistInspector.aliasd`, which has likewise
changed.


`.ConnectionPatch` accepts arbitrary transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively to strings like ``"data"`` or ``"axes fraction"``
`ConnectionPatch` now accepts any `~matplotlib.transforms.Transform`
as input for the ``coordsA`` and ``coordsB`` argument. This allows to
draw lines between points defined in different user defined coordinate
systems. Also see the :doc:`Connect Simple01 example
</gallery/userdemo/connect_simple01>`.


mplot3d Line3D now allows {set,get}_data_3d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lines created with the 3d projection in mplot3d can now access the
data using `~.mplot3d.art3d.Line3D.get_data_3d()` which returns a
tuple of array_likes containing the (x, y, z) data. The equivalent
`~.mplot3d.art3d.Line3D.set_data_3d` can be used to modify the data of
an existing Line3D.


``Axes3D.voxels`` now shades the resulting voxels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.Axes3D.voxels` method now takes a
*shade* parameter that defaults to `True`. This shades faces based
on their orientation, behaving just like the matching parameters to
:meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf` and
:meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.bar3d`.  The plot below shows how
this affects the output.

.. plot::

	import matplotlib.pyplot as plt
	import numpy as np

	# prepare some coordinates
	x, y, z = np.indices((8, 8, 8))

	# draw cuboids in the top left and bottom right corners, and a link between them
	cube1 = (x < 3) & (y < 3) & (z < 3)
	cube2 = (x >= 5) & (y >= 5) & (z >= 5)
	link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

	# combine the objects into a single boolean array
	voxels = cube1 | cube2 | link

	# set the colors of each object
	colors = np.empty(voxels.shape, dtype=object)
	colors[link] = 'red'
	colors[cube1] = 'blue'
	colors[cube2] = 'green'

	# and plot everything
	fig = plt.figure(figsize=plt.figaspect(0.5))
	ax, ax_shaded = fig.subplots(1, 2, subplot_kw=dict(projection='3d'))
	ax.voxels(voxels, facecolors=colors, edgecolor='k', shade=False)
	ax.set_title("Unshaded")
	ax_shaded.voxels(voxels, facecolors=colors, edgecolor='k', shade=True)
	ax_shaded.set_title("Shaded (default)")

	plt.show()

Axis and Ticks
--------------

Added `.Axis.get_inverted` and `.Axis.set_inverted`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `.Axis.get_inverted` and `.Axis.set_inverted` methods query and set whether
the axis uses "inverted" orientation (i.e. increasing to the left for the
x-axis and to the bottom for the y-axis).

They perform tasks similar to `.Axes.xaxis_inverted`,
`.Axes.yaxis_inverted`, `.Axes.invert_xaxis`, and
`.Axes.invert_yaxis`, with the specific difference that
`.Axes..set_inverted` makes it easier to set the invertedness of an
axis regardless of whether it had previously been inverted before.

Adjust default minor tick spacing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default minor tick spacing was changed from 0.625 to 0.5 for major ticks spaced
2.5 units apart.


`.EngFormatter` now accepts *usetex*, *useMathText* as keyword only arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A public API has been added to `.EngFormatter` to control how the
numbers in the ticklabels will be rendered. By default,
``useMathText`` evaluates to
:rc:`axes.formatter.use_mathtext'` and ``usetex`` evaluates
to :rc:`'text.usetex'`.

If either is `True` then the numbers will be encapsulated by ``$``
signs.  When using ``TeX`` this implies that the numbers will be shown
in TeX's math font. When using mathtext, the ``$`` signs around
numbers will ensure unicode rendering (as implied by mathtext). This
will make sure that the minus signs in the ticks are rendered as the
unicode=minus (U+2212) when using mathtext (without relying on the
`~.Formatter.fix_minus` method).



Animation and Interactivity
---------------------------

Support for forward/backward mouse buttons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Figure managers now support a ``button_press`` event for mouse
buttons, similar to the ``key_press`` events. This allows binding
actions to mouse buttons (see `.MouseButton`) The first application of
this mechanism is support of forward/backward mouse buttons in figures
created with the Qt5 backend.


*progress_callback* argument to `~.Animation.save()`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method `.Animation.save` gained an optional
*progress_callback* argument to notify the saving progress.


Add ``cache_frame_data`` keyword-only argument into `.animation.FuncAnimation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.matplotlib.animation.FuncAnimation` has been caching frame data by
default; however, this caching is not ideal in certain cases e.g. When
`.FuncAnimation` needs to be only drawn(not saved) interactively and
memory required by frame data is quite large. By adding
*cache_frame_data* keyword-only argument, users can now disable this
caching; thereby, this new argument provides a fix for issue
:ghissue:`8528`.


Endless Looping GIFs with PillowWriter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We acknowledge that most people want to watch a gif more than
once. Saving an animation as a gif with PillowWriter now produces an
endless looping gif.


Adjusted `.matplotlib.widgets.Slider` to have vertical orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`matplotlib.widgets.Slider` widget now takes an optional
argument *orientation* which indicates the direction
(``'horizontal'`` or ``'vertical'``) that the slider should take.

Improved formatting of image values under cursor when a colorbar is present
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a colorbar is present, its formatter is now used to format the image
values under the mouse cursor in the status bar.  For example, for an image
displaying the values 10,000 and 10,001, the statusbar will now (using default
settings) display the values as ``10000`` and ``10001``), whereas both values
were previously displayed as ``1e+04``.

MouseEvent button attribute is now an IntEnum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :attr:`button` attribute of `~.MouseEvent` instances can take the values
None, 1 (left button), 2 (middle button), 3 (right button), "up" (scroll), and
"down" (scroll).  For better legibility, the 1, 2, and 3 values are now
represented using the `enum.IntEnum` class `matplotlib.backend_bases.MouseButton`,
with the values `.MouseButton.LEFT` (``== 1``), `.MouseButton.MIDDLE` (``== 2``),
and `.MouseButton.RIGHT` (``== 3``).


Configuration, Install, and Development
---------------------------------------

The MATPLOTLIBRC environment variable can now point to any "file" path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This includes device files; in particular, on Unix systems, one can set
``MATPLOTLIBRC`` to ``/dev/null`` to ignore the user's matplotlibrc file and
fall back to Matplotlib's defaults.

As a reminder, if ``MATPLOTLIBRC`` points to a directory, Matplotlib will try
to load the matplotlibrc file from ``$MATPLOTLIBRC/matplotlibrc``.


Allow LaTeX code ``pgf.preamble`` and ``text.latex.preamble`` in MATPLOTLIBRC file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the rc file keys :rc:`pgf.preamble` and
:rc:`text.latex.preamble` were parsed using commas as separators. This
would break valid LaTeX code, such as::

   \usepackage[protrusion=true, expansion=false]{microtype}

The parsing has been modified to pass the complete line to the LaTeX
system, keeping all commas. Passing a list of strings from within a
Python script still works as it used to.



New logging API
~~~~~~~~~~~~~~~

`matplotlib.set_loglevel` / `.pyplot.set_loglevel` can be called to
display more (or less) detailed logging output.
