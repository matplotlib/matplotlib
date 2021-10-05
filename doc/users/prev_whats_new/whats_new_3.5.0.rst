==============================
What's new in Matplotlib 3.5.0
==============================

For a list of all of the issues and pull requests since the last revision, see
the :ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 4

.. toctree::
   :maxdepth: 4

Figure and Axes creation / management
=====================================

Figure now has ``draw_without_rendering`` method
------------------------------------------------

Some aspects of a figure are only determined at draw-time, such as the exact
position of text artists or deferred computation like automatic data limits.
If you need these values, you can use ``figure.canvas.draw()`` to force a full
draw. However, this has side effects, sometimes requires an open file, and is
doing more work than is needed.

The new `.Figure.draw_without_rendering` method runs all the updates that
``draw()`` does, but skips rendering the figure. It's thus more efficient if
you need the updated values to configure further aspects of the figure.

Figure ``__init__`` passes keyword arguments through to set
-----------------------------------------------------------

Similar to many other sub-classes of `~.Artist`, the `~.FigureBase`,
`~.SubFigure`, and `~.Figure` classes will now pass any additional keyword
arguments to `~.Artist.set` to allow properties of the newly created object to
be set at initialization time. For example::

    from matplotlib.figure import Figure
    fig = Figure(label='my figure')

Plotting methods
================

Add ``Annulus`` patch
---------------------

`.Annulus` is a new class for drawing elliptical annuli.

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.patches import Annulus

    fig, ax = plt.subplots()
    cir = Annulus((0.5, 0.5), 0.2, 0.05, fc='g')        # circular annulus
    ell = Annulus((0.5, 0.5), (0.5, 0.3), 0.1, 45,      # elliptical
                  fc='m', ec='b', alpha=0.5, hatch='xxx')
    ax.add_patch(cir)
    ax.add_patch(ell)
    ax.set_aspect('equal')

``set_data`` method for ``FancyArrow`` patch
--------------------------------------------

`.FancyArrow`, the patch returned by ``ax.arrow``, now has a ``set_data``
method that allows modifying the arrow after creation, e.g., for animation.

New arrow styles in ``ArrowStyle`` and ``ConnectionPatch``
----------------------------------------------------------

The new *arrow* parameter to `.ArrowStyle` substitutes the use of the
*beginarrow* and *endarrow* parameters in the creation of arrows. It receives
arrows strings like ``'<-'``, ``']-[``' and ``']->``' instead of individual
booleans.

Two new styles ``']->'`` and ``'<-['`` are also added via this mechanism.
`.ConnectionPatch`, which accepts arrow styles though its *arrowstyle*
parameter, also accepts these new styles.

Setting collection offset transform after initialization
--------------------------------------------------------

The added `.collections.Collection.set_offset_transform` may be used to set the
offset transform after initialization. This can be helpful when creating a
`.collections.Collection` outside an Axes object and later adding it with
`.Axes.add_collection()` and settings the offset transform to
`.Axes.transData`.

Colors and colormaps
====================

Colormap registry (experimental)
--------------------------------

Colormaps are now managed via `matplotlib.colormaps` (or `.pyplot.colormaps`),
which is a `.ColormapRegistry`. While we are confident that the API is final,
we formally mark it as experimental for 3.5 because we want to keep the option
to still adapt the API for 3.6 should the need arise.

Colormaps can be obtained using item access::

    import matplotlib.pyplot as plt
    cmap = plt.colormaps['viridis']

To register new colormaps use::

    plt.colormaps.register(my_colormap)

We recommend to use the new API instead of the `~.cm.get_cmap` and
`~.cm.register_cmap` functions for new code. `matplotlib.cm.get_cmap` and
`matplotlib.cm.register_cmap` will eventually be deprecated and removed.
Within `.pyplot`, ``plt.get_cmap()`` and ``plt.register_cmap()`` will continue
to be supported for backward compatibility.

Image interpolation now possible at RGBA stage
----------------------------------------------

Images in Matplotlib created via `~.axes.Axes.imshow` are resampled to match
the resolution of the current canvas. It is useful to apply an auto-aliasing
filter when downsampling to reduce Moiré effects. By default, interpolation is
done on the data, a norm applied, and then the colormapping performed.

However, it is often desirable for the anti-aliasing interpolation to happen
in RGBA space, where the colors are interpolated rather than the data. This
usually leads to colors outside the colormap, but visually blends adjacent
colors, and is what browsers and other image processing software do.

A new keyword argument *interpolation_stage* is provided for
`~.axes.Axes.imshow` to set the stage at which the anti-aliasing interpolation
happens. The default is the current behaviour of "data", with the alternative
being "rgba" for the newly-available behavior.

For more details see the discussion of the new keyword argument in
:doc:`/gallery/images_contours_and_fields/image_antialiasing`.

A callback registry has been added to Normalize objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.colors.Normalize` objects now have a callback registry, ``callbacks``, that
can be connected to by other objects to be notified when the norm is updated.
The callback emits the key ``changed`` when the norm is modified.
`.cm.ScalarMappable` is now a listener and will register a change when the
norm's vmin, vmax or other attributes are changed.

Titles, ticks, and labels
=========================

Settings tick positions and labels simultaneously in ``set_ticks``
------------------------------------------------------------------

`.Axis.set_ticks` (and the corresponding `.Axes.set_xticks` /
`.Axes.set_yticks`) has a new parameter *labels* allowing to set tick positions
and labels simultaneously.

Previously, setting tick labels was done using `.Axis.set_ticklabels` (or
the corresponding `.Axes.set_xticklabels` / `.Axes.set_yticklabels`). This
usually only makes sense if tick positions were previously fixed with
`~.Axis.set_ticks`. The combined functionality is now available in
`~.Axis.set_ticks`. The use of `.Axis.set_ticklabels` is discouraged, but it
will stay available for backward compatibility.

Note: This addition makes the API of `~.Axis.set_ticks` also more similar to
`.pyplot.xticks` / `.pyplot.yticks`, which already had the additional *labels*
parameter.

Fonts and Text
==============

Font properties of legend title are configurable
------------------------------------------------

Title's font properties can be set via the *title_fontproperties* keyword
argument, for example:

.. plot::

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(10), label='point')
    ax.legend(title='Points',
              title_fontproperties={'family': 'serif', 'size': 20})

Text can be positioned inside TextBox widget
--------------------------------------------

A new parameter called *textalignment* can be used to control for the position
of the text inside the Axes of the `.TextBox` widget.

.. plot::

    from matplotlib import pyplot as plt
    from matplotlib.widgets import TextBox

    for i, alignment in enumerate(['left', 'center', 'right']):
            box_input = plt.axes([0.2, 0.7 - i*0.2, 0.6, 0.1])
            text_box = TextBox(ax=box_input, initial=f'{alignment} alignment',
                               label='', textalignment=alignment)

Simplifying the font setting for usetex mode
--------------------------------------------

Now the :rc:`font.family` accepts some font names as value for a more
user-friendly setup.

.. code-block::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

Type 42 subsetting is now enabled for PDF/PS backends
-----------------------------------------------------

`~matplotlib.backends.backend_pdf` and `~matplotlib.backends.backend_ps` now
use a unified Type 42 font subsetting interface, with the help of `fontTools
<https://fonttools.readthedocs.io/en/latest/>`_

Set :rc:`pdf.fonttype` or :rc:`ps.fonttype` to ``42`` to trigger this
workflow::

    # for PDF backend
    plt.rcParams['pdf.fonttype'] = 42

    # for PS backend
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots()
    ax.text(0.4, 0.5, 'subsetted document is smaller in size!')

    fig.savefig("document.pdf")
    fig.savefig("document.ps")

rcParams improvements
=====================

Allow setting default legend labelcolor globally
------------------------------------------------

A new :rc:`legend.labelcolor` sets the default *labelcolor* argument for
`.Figure.legend`.  The special values  'linecolor', 'markerfacecolor' (or
'mfc'), or 'markeredgecolor' (or 'mec') will cause the legend text to match the
corresponding color of marker.


.. plot::

    plt.rcParams['legend.labelcolor'] = 'linecolor'

    # Make some fake data.
    a = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]

    fig, ax = plt.subplots()
    ax.plot(a, c, 'g--', label='Model length')
    ax.plot(a, d, 'r:', label='Data length')

    ax.legend()

    plt.show()

3D Axes improvements
====================

Axes3D now allows manual control of draw order
----------------------------------------------

The `~mpl_toolkits.mplot3d.axes3d.Axes3D` class now has *computed_zorder*
parameter. When set to False, Artists are drawn using their ``zorder``
attribute.

Allow changing the vertical axis in 3d plots
----------------------------------------------

`~mpl_toolkits.mplot3d.axes3d.Axes3D.view_init` now has the parameter
*vertical_axis* which allows switching which axis is aligned vertically.

Interactive tool improvements
=============================

Updated the appearance of Slider widgets
----------------------------------------

The appearance of `~.Slider` and `~.RangeSlider` widgets were updated and given
new styling parameters for the added handles.

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    plt.figure(figsize=(4, 2))
    ax_old = plt.axes([0.2, 0.65, 0.65, 0.1])
    ax_new = plt.axes([0.2, 0.25, 0.65, 0.1])
    Slider(ax_new, "New", 0, 1)

    ax = ax_old
    valmin = 0
    valinit = 0.5
    ax.set_xlim([0, 1])
    ax_old.axvspan(valmin, valinit, 0, 1)
    ax.axvline(valinit, 0, 1, color="r", lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        -0.02,
        0.5,
        "Old",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="right",
    )

    ax.text(
        1.02,
        0.5,
        "0.5",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="left",
    )

Removing points on a PolygonSelector
------------------------------------

After completing a `~matplotlib.widgets.PolygonSelector`, individual points can
now be removed by right-clicking on them.

Dragging selectors
------------------

The `~matplotlib.widgets.SpanSelector`, `~matplotlib.widgets.RectangleSelector`
and `~matplotlib.widgets.EllipseSelector` have a new keyword argument,
*drag_from_anywhere*, which when set to `True` allows you to click and drag
from anywhere inside the selector to move it. Previously it was only possible
to move it by either activating the move modifier button, or clicking on the
central handle.

The size of the `~matplotlib.widgets.SpanSelector` can now be changed using the
edge handles.

Clearing selectors
------------------

The selectors (`~.widgets.EllipseSelector`, `~.widgets.LassoSelector`,
`~.widgets.PolygonSelector`, `~.widgets.RectangleSelector`, and
`~.widgets.SpanSelector`) have a new method *clear*, which will clear the
current selection and get the selector ready to make a new selection. This is
equivalent to pressing the *escape* key.

Setting artist properties of selectors
--------------------------------------

The artist properties of the `~.widgets.EllipseSelector`,
`~.widgets.LassoSelector`, `~.widgets.PolygonSelector`,
`~.widgets.RectangleSelector` and `~.widgets.SpanSelector` selectors can be
changed using the ``set_props`` and ``set_handle_props`` methods.

Ignore events outside selection
-------------------------------

The `~.widgets.EllipseSelector`, `~.widgets.RectangleSelector` and
`~.widgets.SpanSelector` selectors have a new keyword argument,
*ignore_event_outside*, which when set to `True` will ignore events outside of
the current selection. The handles or the new dragging functionality can instead
be used to change the selection.

``CallbackRegistry`` objects gain a method to temporarily block signals
-----------------------------------------------------------------------

The context manager `~matplotlib.cbook.CallbackRegistry.blocked` can be used
to block callback signals from being processed by the ``CallbackRegistry``.
The optional keyword, *signal*, can be used to block a specific signal
from being processed and let all other signals pass.

.. code-block::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])

    # Block all interactivity through the canvas callbacks
    with fig.canvas.callbacks.blocked():
        plt.show()

    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])

    # Only block key press events
    with fig.canvas.callbacks.blocked(signal="key_press_event"):
        plt.show()

Sphinx extensions
=================

More configuration of ``mathmpl`` sphinx extension
--------------------------------------------------

The `matplotlib.sphinxext.mathmpl` sphinx extension supports two new
configuration options that may be specified in your ``conf.py``:

- ``mathmpl_fontsize`` (float), which sets the font size of the math text in
  points;
- ``mathmpl_srcset`` (list of str), which provides a list of sizes to support
  `responsive resolution images
  <https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images>`__
  The list should contain additional x-descriptors (``'1.5x'``, ``'2x'``, etc.)
  to generate (1x is the default and always included.)

Backend-specific improvements
=============================

Version information
===================

We switched to the `release-branch-semver`_ version scheme. This only affects,
the version information for development builds. Their version number now
describes the targeted release, i.e. 3.5.0.dev820+g6768ef8c4c.d20210520 is 820
commits after the previous release and is scheduled to be officially released
as 3.5.0 later.

In addition to the string ``__version__``, there is now a namedtuple
``__version_info__`` as well, which is modelled after `sys.version_info`_. Its
primary use is safely comparing version information, e.g.  ``if
__version_info__ >= (3, 4, 2)``.

.. _release-branch-semver: https://github.com/pypa/setuptools_scm#version-number-construction
.. _sys.version_info: https://docs.python.org/3/library/sys.html#sys.version_info
