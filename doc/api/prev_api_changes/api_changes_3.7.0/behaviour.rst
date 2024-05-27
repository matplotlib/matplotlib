Behaviour Changes
-----------------

All Axes have ``get_subplotspec`` and ``get_gridspec`` methods now, which returns None for Axes not positioned via a gridspec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, this method was only present for Axes positioned via a gridspec.
Following this change, checking ``hasattr(ax, "get_gridspec")`` should now be
replaced by ``ax.get_gridspec() is not None``.  For compatibility with older
Matplotlib releases, one can also check
``hasattr(ax, "get_gridspec") and ax.get_gridspec() is not None``.

``HostAxesBase.get_aux_axes`` now defaults to using the same base axes class as the host axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using an ``mpl_toolkits.axisartist``-based host Axes, the parasite Axes will
also be based on ``mpl_toolkits.axisartist``.  This behavior is consistent with
``HostAxesBase.twin``, ``HostAxesBase.twinx``, and ``HostAxesBase.twiny``.

``plt.get_cmap`` and ``matplotlib.cm.get_cmap`` return a copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly, `~.pyplot.get_cmap` and ``matplotlib.cm.get_cmap`` returned a global version
of a `.Colormap`. This was prone to errors as modification of the colormap would
propagate from one location to another without warning. Now, a new copy of the colormap
is returned.

``TrapezoidMapTriFinder`` uses different random number generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The random number generator used to determine the order of insertion of
triangle edges in ``TrapezoidMapTriFinder`` has changed. This can result in a
different triangle index being returned for a point that lies exactly on an
edge between two triangles. This can also affect triangulation interpolation
and refinement algorithms that use ``TrapezoidMapTriFinder``.

``FuncAnimation(save_count=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``save_count=None`` to `.FuncAnimation` no longer limits the number
of frames to 100. Make sure that it either can be inferred from *frames*
or provide an integer *save_count*.

``CenteredNorm`` halfrange is not modified when vcenter changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the **halfrange** would expand in proportion to the
amount that **vcenter** was moved away from either **vmin** or **vmax**.
Now, the halfrange remains fixed when vcenter is changed, and **vmin** and
**vmax** are updated based on the **vcenter** and **halfrange** values.

For example, this is what the values were when changing vcenter previously.

.. code-block:: python

    norm = CenteredNorm(vcenter=0, halfrange=1)
    # Move vcenter up by one
    norm.vcenter = 1
    # updates halfrange and vmax (vmin stays the same)
    # norm.halfrange == 2, vmin == -1, vmax == 3

and now, with that same example

.. code-block:: python

    norm = CenteredNorm(vcenter=0, halfrange=1)
    norm.vcenter = 1
    # updates vmin and vmax (halfrange stays the same)
    # norm.halfrange == 1, vmin == 0, vmax == 2

The **halfrange** can be set manually or ``norm.autoscale()``
can be used to automatically set the limits after setting **vcenter**.

``fig.subplot_mosaic`` no longer passes the ``gridspec_kw`` args to nested gridspecs.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nested `.Figure.subplot_mosaic` layouts, it is almost always
inappropriate for *gridspec_kw* arguments to be passed to lower nest
levels, and these arguments are incompatible with the lower levels in
many cases. This dictionary is no longer passed to the inner
layouts. Users who need to modify *gridspec_kw* at multiple levels
should use `.Figure.subfigures` to get nesting, and construct the
inner layouts with `.Figure.subplots` or `.Figure.subplot_mosaic`.

``HPacker`` alignment with **bottom** or **top** are now correct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the **bottom** and **top** alignments were swapped.
This has been corrected so that the alignments correspond appropriately.

On Windows only fonts known to the registry will be discovered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, Matplotlib would recursively walk user and system font directories
to discover fonts, however this lead to a number of undesirable behaviors
including finding deleted fonts. Now Matplotlib will only find fonts that are
known to the Windows registry.

This means that any user installed fonts must go through the Windows font
installer rather than simply being copied to the correct folder.

This only impacts the set of fonts Matplotlib will consider when using
`matplotlib.font_manager.findfont`. To use an arbitrary font, directly pass the
path to a font as shown in
:doc:`/gallery/text_labels_and_annotations/font_file`.

``QuadMesh.set_array`` now always raises ``ValueError`` for inputs with incorrect shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It could previously also raise `TypeError` in some cases.

``contour`` and ``contourf`` auto-select suitable levels when given boolean inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the height array given to `.Axes.contour` or `.Axes.contourf` is of bool
dtype and *levels* is not specified, *levels* now defaults to ``[0.5]`` for
`~.Axes.contour` and ``[0, 0.5, 1]`` for `.Axes.contourf`.

``contour`` no longer warns if no contour lines are drawn.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can occur if the user explicitly passes a ``levels`` array with no values
between ``z.min()`` and ``z.max()``; or if ``z`` has the same value everywhere.

``AxesImage.set_extent`` now raises ``TypeError`` for unknown keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It previously raised a `ValueError`.

Change of ``legend(loc="best")`` behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm of the auto-legend locator has been tweaked to better handle
non rectangular patches. Additional details on this change can be found in
:ghissue:`9580` and :ghissue:`9598`.
