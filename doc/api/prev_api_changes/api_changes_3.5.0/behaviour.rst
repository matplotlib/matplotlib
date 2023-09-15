Behaviour changes
-----------------

First argument to ``subplot_mosaic`` renamed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both `.Figure.subplot_mosaic`, and `.pyplot.subplot_mosaic` have had the
first positional argument renamed from *layout* to *mosaic*. As we have
consolidated the *constrained_layout* and *tight_layout* keyword arguments in
the Figure creation functions of `.pyplot` into a single *layout* keyword
argument, the original ``subplot_mosaic`` argument name would collide.

As this API is provisional, we are changing this argument name with no
deprecation period.

.. _Behavioural API Changes 3.5 - Axes children combined:

``Axes`` children are no longer separated by type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly, `.axes.Axes` children were separated by `.Artist` type, into sublists
such as ``Axes.lines``. For methods that produced multiple elements (such as
`.Axes.errorbar`), though individual parts would have similar *zorder*, this
separation might cause them to be drawn at different times, causing
inconsistent results when overlapping other Artists.

Now, the children are no longer separated by type, and the sublist properties
are generated dynamically when accessed. Consequently, Artists will now always
appear in the correct sublist; e.g., if `.axes.Axes.add_line` is called on a
`.Patch`, it will appear in the ``Axes.patches`` sublist, *not* ``Axes.lines``.
The ``Axes.add_*`` methods will now warn if passed an unexpected type.

Modification of the following sublists is still accepted, but deprecated:

* ``Axes.artists``
* ``Axes.collections``
* ``Axes.images``
* ``Axes.lines``
* ``Axes.patches``
* ``Axes.tables``
* ``Axes.texts``

To remove an Artist, use its `.Artist.remove` method. To add an Artist, use the
corresponding ``Axes.add_*`` method.

``MatplotlibDeprecationWarning`` now subclasses ``DeprecationWarning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Historically, it has not been possible to filter
`~matplotlib.MatplotlibDeprecationWarning`\s by checking for
`DeprecationWarning`, since we subclass `UserWarning` directly.

The decision to not subclass `DeprecationWarning` has to do with a decision
from core Python in the 2.x days to not show `DeprecationWarning`\s to users.
However, there is now a more sophisticated filter in place (see
https://www.python.org/dev/peps/pep-0565/).

Users will now see `~matplotlib.MatplotlibDeprecationWarning` only during
interactive sessions, and these can be silenced by the standard mechanism:

.. code:: python

	warnings.filterwarnings("ignore", category=DeprecationWarning)

Library authors must now enable `DeprecationWarning`\s explicitly in order for
(non-interactive) CI/CD pipelines to report back these warnings, as is standard
for the rest of the Python ecosystem:

.. code:: python

	warnings.filterwarnings("always", DeprecationWarning)

``Artist.set`` applies artist properties in the order in which they are given
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The change only affects the interaction between the *color*, *edgecolor*,
*facecolor*, and (for `.Collection`\s) *alpha* properties: the *color* property
now needs to be passed first in order not to override the other properties.
This is consistent with e.g. `.Artist.update`, which did not reorder the
properties passed to it.

``pcolor(mesh)`` shading defaults to auto
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *shading* keyword argument for `.Axes.pcolormesh` and `.Axes.pcolor`
default has been changed to 'auto'.

Passing ``Z(M, N)``, ``x(N)``, ``y(M)`` to ``pcolormesh`` with
``shading='flat'`` will now raise a `TypeError`. Use ``shading='auto'`` or
``shading='nearest'`` for ``x`` and ``y`` to be treated as cell centers, or
drop the last column and row of ``Z`` to get the old behaviour with
``shading='flat'``.

Colorbars now have pan and zoom functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive plots with colorbars can now be zoomed and panned on the colorbar
axis. This adjusts the *vmin* and *vmax* of the `.ScalarMappable` associated
with the colorbar. This is currently only enabled for continuous norms. Norms
used with ``contourf`` and categoricals, such as `.BoundaryNorm` and `.NoNorm`,
have the interactive capability disabled by default. `cb.ax.set_navigate()
<.Axes.set_navigate>` can be used to set whether a colorbar axes is interactive
or not.

Colorbar lines no longer clipped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a colorbar has lines added to it (e.g. for contour lines), these will no
longer be clipped. This is an improvement for lines on the edge of the
colorbar, but could lead to lines off the colorbar if the limits of the
colorbar are changed.

``Figure.suppressComposite`` now also controls compositing of Axes images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output of ``NonUniformImage`` and ``PcolorImage`` has changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pixel-level differences may be observed in images generated using
`.NonUniformImage` or `.PcolorImage`, typically for pixels exactly at the
boundary between two data cells (no user-facing axes method currently generates
`.NonUniformImage`\s, and only `.pcolorfast` can generate `.PcolorImage`\s).
These artists are also now slower, normally by ~1.5x but sometimes more (in
particular for ``NonUniformImage(interpolation="bilinear")``. This slowdown
arises from fixing occasional floating point inaccuracies.

Change of the (default) legend handler for ``Line2D`` instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default legend handler for Line2D instances (`.HandlerLine2D`) now
consistently exposes all the attributes and methods related to the line marker
(:ghissue:`11358`). This makes it easy to change the marker features after
instantiating a legend.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot([1, 3, 2], marker="s", label="Line", color="pink", mec="red", ms=8)
    leg = ax.legend()

    leg.legendHandles[0].set_color("lightgray")
    leg.legendHandles[0].set_mec("black")  # marker edge color

The former legend handler for Line2D objects has been renamed
`.HandlerLine2DCompound`. To revert to the previous behaviour, one can use

.. code-block:: python

    import matplotlib.legend as mlegend
    from matplotlib.legend_handler import HandlerLine2DCompound
    from matplotlib.lines import Line2D

    mlegend.Legend.update_default_handler_map({Line2D: HandlerLine2DCompound()})

Setting ``Line2D`` marker edge/face color to *None* use rcParams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Line2D.set_markeredgecolor(None)`` and ``Line2D.set_markerfacecolor(None)``
now set the line property using the corresponding rcParam
(:rc:`lines.markeredgecolor` and :rc:`lines.markerfacecolor`). This is
consistent with other `.Line2D` property setters.

Default theta tick locations for wedge polar plots have changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For polar plots that don't cover a full circle, the default theta tick
locations are now at multiples of 10°, 15°, 30°, 45°, 90°, rather than using
values that mostly make sense for linear plots (20°, 25°, etc.).

``axvspan`` now plots full wedges in polar plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... rather than triangles.

Convenience converter from ``Scale`` to ``Normalize`` now public
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Downstream libraries can take advantage of `.colors.make_norm_from_scale` to
create a `~.colors.Normalize` subclass directly from an existing scale.
Usually norms have a scale, and the advantage of having a  `~.scale.ScaleBase`
attached to a norm is to provide a scale, and associated tick locators and
formatters, for the colorbar.

``ContourSet`` always use ``PathCollection``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to correct rendering issues with closed loops, the `.ContourSet` now
creates a `.PathCollection` instead of a `.LineCollection` for line contours.
This type matches the artist used for filled contours.

This affects `.ContourSet` itself and its subclasses, `.QuadContourSet`
(returned by `.Axes.contour`), and `.TriContourSet` (returned by
`.Axes.tricontour`).

``hatch.SmallFilledCircles`` inherits from ``hatch.Circles``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.hatch.SmallFilledCircles` class now inherits from `.hatch.Circles` rather
than from `.hatch.SmallCircles`.

hexbin with a log norm
~~~~~~~~~~~~~~~~~~~~~~

`~.axes.Axes.hexbin` no longer (incorrectly) adds 1 to every bin value if a log
norm is being used.

Setting invalid ``rcParams["date.converter"]`` now raises ValueError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, invalid values passed to :rc:`date.converter` would be ignored with
a `UserWarning`, but now raise `ValueError`.

``Text`` and ``TextBox`` added *parse_math* option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.Text` and `.TextBox` objects now allow a *parse_math* keyword-only argument
which controls whether math should be parsed from the displayed string. If
*True*, the string will be parsed as a math text object. If *False*, the string
will be considered a literal and no parsing will occur.

For `.Text`, this argument defaults to *True*. For `.TextBox` this argument
defaults to *False*.

``Type1Font`` objects now decrypt the encrypted part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Type 1 fonts have a large part of their code encrypted as an obsolete
copy-protection measure. This part is now available decrypted as the
``decrypted`` attribute of ``matplotlib.type1font.Type1Font``. This decrypted
data is not yet parsed, but this is a prerequisite for implementing subsetting.

3D contourf polygons placed between levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The polygons used in a 3D `~.Axes3D.contourf` plot are now
placed halfway between the contour levels, as each polygon represents the
location of values that lie between two levels.

``AxesDivider`` now defaults to rcParams-specified pads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.AxesDivider.append_axes`, ``AxesDivider.new_horizontal``, and
``AxesDivider.new_vertical`` now default to paddings specified by
:rc:`figure.subplot.wspace` and :rc:`figure.subplot.hspace` rather than zero.
