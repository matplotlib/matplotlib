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

``subplot_mosaic`` supports simple Axes sharing
-----------------------------------------------

`.Figure.subplot_mosaic`, `.pyplot.subplot_mosaic` support *simple* Axes
sharing (i.e., only `True`/`False` may be passed to *sharex*/*sharey*). When
`True`, tick label visibility and Axis units will be shared.

.. plot::
    :include-source:

    mosaic = [
        ['A', [['B', 'C'],
               ['D', 'E']]],
        ['F', 'G'],
    ]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic, sharex=True, sharey=True)
    # All Axes use these scales after this call.
    ax_dict['A'].set(xscale='log', yscale='logit')

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

.. plot::

    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot([0.75, 0.75], [0.25, 0.75], 'ok')
    ax.set(xlim=(0, 1), ylim=(0, 1), title='New ArrowStyle options')

    ax.annotate(']->', (0.75, 0.25), (0.25, 0.25),
                arrowprops=dict(
                    arrowstyle=']->', connectionstyle="arc3,rad=-0.05",
                    shrinkA=5, shrinkB=5,
                ),
                bbox=dict(boxstyle='square', fc='w'), size='large')

    ax.annotate('<-[', (0.75, 0.75), (0.25, 0.75),
                arrowprops=dict(
                    arrowstyle='<-[', connectionstyle="arc3,rad=-0.05",
                    shrinkA=5, shrinkB=5,
                ),
                bbox=dict(boxstyle='square', fc='w'), size='large')

Setting collection offset transform after initialization
--------------------------------------------------------

The added `.collections.Collection.set_offset_transform` may be used to set the
offset transform after initialization. This can be helpful when creating a
`.collections.Collection` outside an Axes object, and later adding it with
`.Axes.add_collection()` and setting the offset transform to `.Axes.transData`.

Colors and colormaps
====================

Colormap registry (experimental)
--------------------------------

Colormaps are now managed via `matplotlib.colormaps` (or `.pyplot.colormaps`),
which is a `.ColormapRegistry`. While we are confident that the API is final,
we formally mark it as experimental for 3.5 because we want to keep the option
to still modify the API for 3.6 should the need arise.

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

.. figure:: /gallery/images_contours_and_fields/images/sphx_glr_image_antialiasing_001.png
   :target: /gallery/images_contours_and_fields/image_antialiasing.html

   Example of the interpolation stage options.

For more details see the discussion of the new keyword argument in
:doc:`/gallery/images_contours_and_fields/image_antialiasing`.

``imshow`` supports half-float arrays
-------------------------------------

The `~.axes.Axes.imshow` method now supports half-float arrays, i.e., NumPy
arrays with dtype ``np.float16``.

A callback registry has been added to Normalize objects
-------------------------------------------------------

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
the corresponding `.Axes.set_xticklabels` / `.Axes.set_yticklabels`); this
usually only makes sense if tick positions were previously fixed with
`~.Axis.set_ticks`::

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['a', 'b', 'c'])

The combined functionality is now available in `~.Axis.set_ticks`::

    ax.set_xticks([1, 2, 3], ['a', 'b', 'c'])

The use of `.Axis.set_ticklabels` is discouraged, but it will stay available
for backward compatibility.

Note: This addition makes the API of `~.Axis.set_ticks` also more similar to
`.pyplot.xticks` / `.pyplot.yticks`, which already had the additional *labels*
parameter.

Fonts and Text
==============

Triple and quadruple dot mathtext accents
-----------------------------------------

In addition to single and double dot accents, mathtext now supports triple and
quadruple dot accents.

.. plot::
    :include-source:

    fig = plt.figure(figsize=(3, 1))
    fig.text(0.5, 0.5, r'$\dot{a} \ddot{b} \dddot{c} \ddddot{d}$', fontsize=40,
             horizontalalignment='center', verticalalignment='center')

Font properties of legend title are configurable
------------------------------------------------

Title's font properties can be set via the *title_fontproperties* keyword
argument, for example:

.. plot::

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(10), label='point')
    ax.legend(title='Points',
              title_fontproperties={'family': 'serif', 'size': 20})

``Text`` and ``TextBox`` added *parse_math* option
--------------------------------------------------

`.Text` and `.TextBox` objects now allow a *parse_math* keyword-only argument
which controls whether math should be parsed from the displayed string. If
*True*, the string will be parsed as a math text object. If *False*, the string
will be considered a literal and no parsing will occur.

Text can be positioned inside TextBox widget
--------------------------------------------

A new parameter called *textalignment* can be used to control for the position
of the text inside the Axes of the `.TextBox` widget.

.. plot::

    from matplotlib import pyplot as plt
    from matplotlib.widgets import TextBox

    fig = plt.figure(figsize=(4, 3))
    for i, alignment in enumerate(['left', 'center', 'right']):
            box_input = fig.add_axes([0.1, 0.7 - i*0.3, 0.8, 0.2])
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

.. plot::

    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import art3d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3),
                                   subplot_kw=dict(projection='3d'))

    ax1.set_title('computed_zorder = True (default)')
    ax2.set_title('computed_zorder = False')
    ax2.computed_zorder = False

    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        tri = art3d.Poly3DCollection([corners],
                                     facecolors='white',
                                     edgecolors='black',
                                     zorder=1)
        ax.add_collection3d(tri)
        line, = ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2,
                        label='zorder=2')
        points = ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10,
                            label='zorder=10')

        ax.set_xlim((0, 5))
        ax.set_ylim((0, 5))
        ax.set_zlim((0, 2.5))

    plane = mpatches.Patch(facecolor='white', edgecolor='black',
                           label='zorder=1')
    fig.legend(handles=[plane, line, points], loc='lower center')

Allow changing the vertical axis in 3d plots
----------------------------------------------

`~mpl_toolkits.mplot3d.axes3d.Axes3D.view_init` now has the parameter
*vertical_axis* which allows switching which axis is aligned vertically.

.. plot::

    Nphi, Nr = 18, 8
    phi = np.linspace(0, np.pi, Nphi)
    r = np.arange(Nr)
    phi = np.tile(phi, Nr).flatten()
    r = np.repeat(r, Nphi).flatten()

    x = r * np.sin(phi)
    y = r * np.cos(phi)
    z = Nr - r

    fig, axs = plt.subplots(1, 3, figsize=(7, 3),
                            subplot_kw=dict(projection='3d'),
                            gridspec_kw=dict(wspace=0.4, left=0.08, right=0.98,
                                             bottom=0, top=1))
    for vert_a, ax in zip(['z', 'y', 'x'], axs):
        pc = ax.scatter(x, y, z, c=z)
        ax.view_init(azim=30, elev=30, vertical_axis=vert_a)
        ax.set(xlabel='x', ylabel='y', zlabel='z',
               title=f'vertical_axis={vert_a!r}')

``plot_surface`` supports masked arrays and NaNs
------------------------------------------------

`.axes3d.Axes3D.plot_surface` supports masked arrays and NaNs, and will now
hide quads that contain masked or NaN points. The behaviour is similar to
`.Axes.contour` with ``corner_mask=True``.

.. plot::

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'},
                           constrained_layout=True)

    x, y = np.mgrid[1:10:1, 1:10:1]
    z = x ** 3 + y ** 3 - 500
    z = np.ma.masked_array(z, z < 0)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, cmap='inferno')
    ax.view_init(35, -90)

3D plotting methods support *data* keyword argument
---------------------------------------------------

To match all 2D plotting methods, the 3D Axes now support the *data* keyword
argument. This allows passing arguments indirectly from a DataFrame-like
structure. ::

    data = {  # A labelled data set, or e.g., Pandas DataFrame.
        'x': ...,
        'y': ...,
        'z': ...,
        'width': ...,
        'depth': ...,
        'top': ...,
    }

    fig, ax = plt.subplots(subplot_kw={'projection': '3d')
    ax.bar3d('x', 'y', 'z', 'width', 'depth', 'top', data=data)

Interactive tool improvements
=============================

Colorbars now have pan and zoom functionality
---------------------------------------------

Interactive plots with colorbars can now be zoomed and panned on the colorbar
axis. This adjusts the *vmin* and *vmax* of the ``ScalarMappable`` associated
with the colorbar. This is currently only enabled for continuous norms. Norms
used with contourf and categoricals, such as ``BoundaryNorm`` and ``NoNorm``,
have the interactive capability disabled by default. ``cb.ax.set_navigate()``
can be used to set whether a colorbar axes is interactive or not.

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

Directional sizing cursors
--------------------------

Canvases now support setting directional sizing cursors, i.e., horizontal and
vertical double arrows. These are used in e.g., selector widgets. Try the
:doc:`/gallery/widgets/mouse_cursor` example to see the cursor in your desired
backend.

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

GTK backend
-----------

A backend supporting GTK4_ has been added. Both Agg and Cairo renderers are
supported. The GTK4 backends may be selected as GTK4Agg or GTK4Cairo.

.. _GTK4: https://www.gtk.org/

Qt backends
-----------

Support for Qt6 (using either PyQt6_ or PySide6_) has been added, with either
the Agg or Cairo renderers. Simultaneously, support for Qt4 has been dropped.
Both Qt6 and Qt5 are supported by a combined backend (QtAgg or QtCairo), and
the loaded version is determined by modules already imported, the
:envvar:`QT_API` environment variable, and available packages. See
:ref:`QT_API-usage` for details. The versioned Qt5 backend names (Qt5Agg or
Qt5Cairo) remain supported for backwards compatibility.

.. _PyQt6: https://www.riverbankcomputing.com/static/Docs/PyQt6/
.. _PySide6: https://doc.qt.io/qtforpython/

HiDPI support in Cairo-based, GTK, and Tk backends
--------------------------------------------------

The GTK3 backends now support HiDPI fully, including mixed monitor cases (on
Wayland only). The newly added GTK4 backends also support HiDPI.

The TkAgg backend now supports HiDPI **on Windows only**, including mixed
monitor cases.

All Cairo-based backends correctly support HiDPI as well as their Agg
counterparts did (i.e., if the toolkit supports HiDPI, then the \*Cairo backend
will now support it, but not otherwise.)

Qt figure options editor improvements
-------------------------------------

The figure options editor in the Qt backend now also supports editing the left
and right titles (plus the existing centre title). Editing Axis limits is
better supported when using a date converter. The ``symlog`` option is now
available in Axis scaling options. All entries with the same label are now
shown in the Curves tab.

WX backends support mouse navigation buttons
--------------------------------------------

The WX backends now support navigating through view states using the mouse
forward/backward buttons, as in other backends.

WebAgg uses asyncio instead of Tornado
--------------------------------------

The WebAgg backend defaults to using `asyncio` over Tornado for timer support.
This allows using the WebAgg backend in JupyterLite.

Version information
===================

We switched to the `release-branch-semver`_ version scheme of setuptools-scm.
This only affects the version information for development builds. Their version
number now describes the targeted release, i.e. 3.5.0.dev820+g6768ef8c4c is 820
commits after the previous release and is scheduled to be officially released
as 3.5.0 later.

In addition to the string ``__version__``, there is now a namedtuple
``__version_info__`` as well, which is modelled after `sys.version_info`_. Its
primary use is safely comparing version information, e.g.  ``if
__version_info__ >= (3, 4, 2)``.

.. _release-branch-semver: https://github.com/pypa/setuptools_scm#version-number-construction
.. _sys.version_info: https://docs.python.org/3/library/sys.html#sys.version_info
