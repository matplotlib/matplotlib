
Changes in 1.5.0
================

Code Changes
------------

Reversed `matplotlib.cbook.ls_mapper`, added `.ls_mapper_r`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly, `matplotlib.cbook.ls_mapper` was a dictionary with
the long-form line-style names (``"solid"``) as keys and the short
forms (``"-"``) as values.  This long-to-short mapping is now done
by `.ls_mapper_r`, and the short-to-long mapping is done by the
`.ls_mapper`.

Prevent moving artists between Axes, Property-ify Artist.axes, deprecate Artist.{get,set}_axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This was done to prevent an Artist that is
already associated with an Axes from being moved/added to a different Axes.
This was never supported as it causes havoc with the transform stack.
The apparent support for this (as it did not raise an exception) was
the source of multiple bug reports and questions on SO.

For almost all use-cases, the assignment of the axes to an artist should be
taken care of by the axes as part of the ``Axes.add_*`` method, hence the
deprecation of {get,set}_axes.

Removing the ``set_axes`` method will also remove the 'axes' line from
the ACCEPTS kwarg tables (assuming that the removal date gets here
before that gets overhauled).

Tightened input validation on 'pivot' kwarg to quiver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tightened validation so that only {'tip', 'tail', 'mid', and 'middle'} (but any
capitalization) are valid values for the *pivot* keyword argument in the
`.Quiver` class (and hence `.axes.Axes.quiver` and `.pyplot.quiver` which both
fully delegate to `.Quiver`).  Previously any input matching 'mid.*' would be
interpreted as 'middle', 'tip.*' as 'tip' and any string not matching one of
those patterns as 'tail'.

The value of `.Quiver.pivot` is normalized to be in the set {'tip', 'tail',
'middle'} in `.Quiver`.

Reordered ``Axes.get_children``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The artist order returned by `.axes.Axes.get_children` did not
match the one used by `.axes.Axes.draw`.  They now use the same
order, as `.axes.Axes.draw` now calls `.axes.Axes.get_children`.

Changed behaviour of contour plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default behaviour of :func:`~matplotlib.pyplot.contour` and
:func:`~matplotlib.pyplot.contourf` when using a masked array is now determined
by the new keyword argument *corner_mask*, or if this is not specified then
the new :rc:`contour.corner_mask` instead.  The new default behaviour is
equivalent to using ``corner_mask=True``; the previous behaviour can be obtained
using ``corner_mask=False`` or by changing the rcParam.  The example
http://matplotlib.org/examples/pylab_examples/contour_corner_mask.html
demonstrates the difference.  Use of the old contouring algorithm, which is
obtained with ``corner_mask='legacy'``, is now deprecated.

Contour labels may now appear in different places than in earlier versions of
Matplotlib.

In addition, the keyword argument *nchunk* now applies to
:func:`~matplotlib.pyplot.contour` as well as
:func:`~matplotlib.pyplot.contourf`, and it subdivides the domain into
subdomains of exactly *nchunk* by *nchunk* quads, whereas previously it was
only roughly *nchunk* by *nchunk* quads.

The C/C++ object that performs contour calculations used to be stored in the
public attribute ``QuadContourSet.Cntr``, but is now stored in a private
attribute and should not be accessed by end users.

Added set_params function to all Locator types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This was a bug fix targeted at making the api for Locators more consistent.

In the old behavior, only locators of type MaxNLocator have set_params()
defined, causing its use on any other Locator to raise an AttributeError *(
aside: set_params(args) is a function that sets the parameters of a Locator
instance to be as specified within args)*. The fix involves moving set_params()
to the Locator class such that all subtypes will have this function defined.

Since each of the Locator subtypes have their own modifiable parameters, a
universal set_params() in Locator isn't ideal. Instead, a default no-operation
function that raises a warning is implemented in Locator. Subtypes extending
Locator will then override with their own implementations. Subtypes that do
not have a need for set_params() will fall back onto their parent's
implementation, which raises a warning as intended.

In the new behavior, Locator instances will not raise an AttributeError
when set_params() is called. For Locators that do not implement set_params(),
the default implementation in Locator is used.

Disallow ``None`` as x or y value in ax.plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not allow ``None`` as a valid input for the ``x`` or ``y`` args in
`.axes.Axes.plot`.  This may break some user code, but this was never
officially supported (ex documented) and allowing ``None`` objects through can
lead to confusing exceptions downstream.

To create an empty line use ::

  ln1, = ax.plot([], [], ...)
  ln2, = ax.plot([], ...)

In either case to update the data in the `.Line2D` object you must update
both the ``x`` and ``y`` data.


Removed *args* and *kwargs* from `.MicrosecondLocator.__call__`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The call signature of :meth:`~matplotlib.dates.MicrosecondLocator.__call__`
has changed from ``__call__(self, *args, **kwargs)`` to ``__call__(self)``.
This is consistent with the superclass :class:`~matplotlib.ticker.Locator`
and also all the other Locators derived from this superclass.


No `ValueError` for the MicrosecondLocator and YearLocator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~matplotlib.dates.MicrosecondLocator` and
:class:`~matplotlib.dates.YearLocator` objects when called will return
an empty list if the axes have no data or the view has no interval.
Previously, they raised a `ValueError`. This is consistent with all
the Date Locators.

'OffsetBox.DrawingArea' respects the 'clip' keyword argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The call signature was ``OffsetBox.DrawingArea(..., clip=True)`` but nothing
was done with the *clip* argument. The object did not do any clipping
regardless of that parameter. Now the object can and does clip the
child `.Artist`\ s if they are set to be clipped.

You can turn off the clipping on a per-child basis using
``child.set_clip_on(False)``.

Add salt to clipPath id
~~~~~~~~~~~~~~~~~~~~~~~

Add salt to the hash used to determine the id of the ``clipPath``
nodes.  This is to avoid conflicts when two svg documents with the same
clip path are included in the same document (see
https://github.com/ipython/ipython/issues/8133 and
https://github.com/matplotlib/matplotlib/issues/4349 ), however this
means that the svg output is no longer deterministic if the same
figure is saved twice.  It is not expected that this will affect any
users as the current ids are generated from an md5 hash of properties
of the clip path and any user would have a very difficult time
anticipating the value of the id.

Changed snap threshold for circle markers to inf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When drawing circle markers above some marker size (previously 6.0)
the path used to generate the marker was snapped to pixel centers.  However,
this ends up distorting the marker away from a circle.  By setting the
snap threshold to inf snapping is never done on circles.

This change broke several tests, but is an improvement.

Preserve units with Text position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously the 'get_position' method on Text would strip away unit information
even though the units were still present.  There was no inherent need to do
this, so it has been changed so that unit data (if present) will be preserved.
Essentially a call to 'get_position' will return the exact value from a call to
'set_position'.

If you wish to get the old behaviour, then you can use the new method called
'get_unitless_position'.

New API for custom Axes view changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive pan and zoom were previously implemented using a Cartesian-specific
algorithm that was not necessarily applicable to custom Axes. Three new private
methods, ``matplotlib.axes._base._AxesBase._get_view``,
``matplotlib.axes._base._AxesBase._set_view``, and
``matplotlib.axes._base._AxesBase._set_view_from_bbox``, allow for custom
*Axes* classes to override the pan and zoom algorithms. Implementors of
custom *Axes* who override these methods may provide suitable behaviour for
both pan and zoom as well as the view navigation buttons on the interactive
toolbars.

MathTex visual changes
----------------------

The spacing commands in mathtext have been changed to more closely
match vanilla TeX.


Improved spacing in mathtext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extra space that appeared after subscripts and superscripts has
been removed.

No annotation coordinates wrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In #2351 for 1.4.0 the behavior of ['axes points', 'axes pixel',
'figure points', 'figure pixel'] as coordinates was change to
no longer wrap for negative values.  In 1.4.3 this change was
reverted for 'axes points' and 'axes pixel' and in addition caused
'axes fraction' to wrap.  For 1.5 the behavior has been reverted to
as it was in 1.4.0-1.4.2, no wrapping for any type of coordinate.

Deprecation
-----------

Deprecated ``GraphicsContextBase.set_graylevel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GraphicsContextBase.set_graylevel`` function has been deprecated in 1.5
and will be removed in 1.6.  It has been unused.  The
`.GraphicsContextBase.set_foreground` could be used instead.

deprecated idle_event
~~~~~~~~~~~~~~~~~~~~~

The ``idle_event`` was broken or missing in most backends and causes spurious
warnings in some cases, and its use in creating animations is now obsolete due
to the animations module. Therefore code involving it has been removed from all
but the wx backend (where it partially works), and its use is deprecated.  The
`.animation` module may be used instead to create animations.

``color_cycle`` deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~

In light of the new property cycling feature,
the Axes method ``set_color_cycle`` is now deprecated.
Calling this method will replace the current property cycle with
one that cycles just the given colors.

Similarly, the rc parameter *axes.color_cycle* is also deprecated in
lieu of the new :rc:`axes.prop_cycle` parameter. Having both parameters in
the same rc file is not recommended as the result cannot be
predicted. For compatibility, setting *axes.color_cycle* will
replace the cycler in :rc:`axes.prop_cycle` with a color cycle.
Accessing *axes.color_cycle* will return just the color portion
of the property cycle, if it exists.

Timeline for removal has not been set.


Bundled jquery
--------------

The version of jquery bundled with the webagg backend has been upgraded
from 1.7.1 to 1.11.3.  If you are using the version of jquery bundled
with webagg you will need to update your html files as such

.. code-block:: diff

   -    <script src="_static/jquery/js/jquery-1.7.1.min.js"></script>
   +    <script src="_static/jquery/js/jquery-1.11.3.min.js"></script>


Code Removed
------------

Removed ``Image`` from main namespace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Image`` was imported from PIL/pillow to test if PIL is available, but
there is no reason to keep ``Image`` in the namespace once the availability
has been determined.

Removed ``lod`` from Artist
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Removed the method ``set_lod`` and all references to the attribute ``_lod`` as
they are not used anywhere else in the code base.  It appears to be a feature
stub that was never built out.

Removed threading related classes from cbook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The classes ``Scheduler``, ``Timeout``, and ``Idle`` were in cbook, but
are not used internally.  They appear to be a prototype for the idle event
system which was not working and has recently been pulled out.

Removed *Lena* images from sample_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lena.png`` and ``lena.jpg`` images have been removed from
Matplotlib's sample_data directory. The images are also no longer
available from `matplotlib.cbook.get_sample_data`. We suggest using
``matplotlib.cbook.get_sample_data('grace_hopper.png')`` or
``matplotlib.cbook.get_sample_data('grace_hopper.jpg')`` instead.


Legend
~~~~~~
Removed handling of *loc* as a positional argument to `.Legend`


Legend handlers
~~~~~~~~~~~~~~~
Remove code to allow legend handlers to be callable.  They must now
implement a method ``legend_artist``.


Axis
~~~~
Removed method ``set_scale``.  This is now handled via a private method which
should not be used directly by users.  It is called via ``Axes.set_{x,y}scale``
which takes care of ensuring the related changes are also made to the Axes
object.

finance.py
~~~~~~~~~~

Removed functions with ambiguous argument order from finance.py


Annotation
~~~~~~~~~~

Removed ``textcoords`` and ``xytext`` proprieties from Annotation objects.


sphinxext.ipython_*.py
~~~~~~~~~~~~~~~~~~~~~~

Both ``ipython_console_highlighting`` and ``ipython_directive`` have been
moved to IPython.

Change your import from ``matplotlib.sphinxext.ipython_directive`` to
``IPython.sphinxext.ipython_directive`` and from
``matplotlib.sphinxext.ipython_directive`` to
``IPython.sphinxext.ipython_directive``


LineCollection.color
~~~~~~~~~~~~~~~~~~~~

Deprecated in 2005, use ``set_color``


remove ``'faceted'`` as a valid value for *shading* in ``tri.tripcolor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use *edgecolor* instead.  Added validation on *shading* to only be valid
values.


Remove ``faceted`` kwarg from scatter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove support for the ``faceted`` kwarg.  This was deprecated in
d48b34288e9651ff95c3b8a071ef5ac5cf50bae7 (2008-04-18!) and replaced by
``edgecolor``.


Remove ``set_colorbar`` method from ``ScalarMappable``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ``set_colorbar`` method, use `~.cm.ScalarMappable.colorbar` attribute
directly.


patheffects.svg
~~~~~~~~~~~~~~~

 - remove ``get_proxy_renderer`` method from ``AbstarctPathEffect`` class
 - remove ``patch_alpha`` and ``offset_xy`` from ``SimplePatchShadow``


Remove ``testing.image_util.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Contained only a no-longer used port of functionality from PIL


Remove ``mlab.FIFOBuffer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Not used internally and not part of core mission of mpl.


Remove ``mlab.prepca``
~~~~~~~~~~~~~~~~~~~~~~
Deprecated in 2009.


Remove ``NavigationToolbar2QTAgg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Added no functionality over the base ``NavigationToolbar2Qt``


mpl.py
~~~~~~

Remove the module ``matplotlib.mpl``.  Deprecated in 1.3 by
PR #1670 and commit 78ce67d161625833cacff23cfe5d74920248c5b2
