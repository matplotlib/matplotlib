
API Changes in 2.0.0
====================

Deprecation and removal
-----------------------

Color of Axes
~~~~~~~~~~~~~
The ``axisbg`` and ``axis_bgcolor`` properties on *Axes* have been
deprecated in favor of ``facecolor``.

GTK and GDK backends deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The GDK and GTK backends have been deprecated. These obsolete backends
allow figures to be rendered via the GDK API to files and GTK2 figures.
They are untested and known to be broken, and their use has been
discouraged for some time.  Instead, use the ``GTKAgg`` and ``GTKCairo``
backends for rendering to GTK2 windows.

WX backend deprecated
~~~~~~~~~~~~~~~~~~~~~
The WX backend has been deprecated.  It is untested, and its
use has been discouraged for some time. Instead, use the ``WXAgg``
backend for rendering figures to WX windows.

CocoaAgg backend removed
~~~~~~~~~~~~~~~~~~~~~~~~
The deprecated and not fully functional CocoaAgg backend has been removed.

`round` removed from TkAgg Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The TkAgg backend had its own implementation of the `round` function. This
was unused internally and has been removed. Instead, use either the
`round` builtin function or `numpy.around`.

.. _v200_deprecate_hold:

'hold' functionality deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The 'hold' keyword argument and all functions and methods related
to it are deprecated, along with the ``axes.hold`` rcParams entry.
The behavior will remain consistent with the default ``hold=True``
state that has long been in place.  Instead of using a function
or keyword argument (``hold=False``) to change that behavior,
explicitly clear the axes or figure as needed prior to subsequent
plotting commands.


`.Artist.update` has return value
---------------------------------

The methods `matplotlib.artist.Artist.set`, `matplotlib.artist.Artist.update`,
and the function `matplotlib.artist.setp` now use a common codepath to look up
how to update the given artist properties (either using the setter methods or
an attribute/property).

The behavior of `matplotlib.artist.Artist.update` is slightly changed to return
a list of the values returned from the setter methods to avoid changing the API
of `matplotlib.artist.Artist.set` and `matplotlib.artist.setp`.

The keys passed into `matplotlib.artist.Artist.update` are now converted to
lower case before being processed, to match the behavior of
`matplotlib.artist.Artist.set` and `matplotlib.artist.setp`.  This should not
break any user code because there are no set methods with capitals in
their names, but this puts a constraint on naming properties in the future.


`.Legend` initializers gain *edgecolor* and *facecolor* keyword arguments
-------------------------------------------------------------------------

The :class:`~matplotlib.legend.Legend` background patch (or 'frame')
can have its ``edgecolor`` and ``facecolor`` determined by the
corresponding keyword arguments to the :class:`matplotlib.legend.Legend`
initializer, or to any of the methods or functions that call that
initializer.  If left to their default values of `None`, their values
will be taken from ``matplotlib.rcParams``.  The previously-existing
``framealpha`` kwarg still controls the alpha transparency of the
patch.


Qualitative colormaps
---------------------

Colorbrewer's qualitative/discrete colormaps ("Accent", "Dark2", "Paired",
"Pastel1", "Pastel2", "Set1", "Set2", "Set3") are now implemented as
`.ListedColormap` instead of `.LinearSegmentedColormap`.

To use these for images where categories are specified as integers, for
instance, use::

    plt.imshow(x, cmap='Dark2', norm=colors.NoNorm())


Change in the ``draw_image`` backend API
----------------------------------------

The ``draw_image`` method implemented by backends has changed its interface.

This change is only relevant if the backend declares that it is able
to transform images by returning ``True`` from ``option_scale_image``.
See the ``draw_image`` docstring for more information.



``matplotlib.ticker.LinearLocator`` algorithm update
----------------------------------------------------

The `matplotlib.ticker.LinearLocator` is used to define the range and
location of axis ticks when the user wants an exact number of ticks.
``LinearLocator`` thus differs from the default locator ``MaxNLocator``,
for which the user specifies a maximum number of intervals rather than
a precise number of ticks.

The view range algorithm in ``matplotlib.ticker.LinearLocator`` has been
changed so that more convenient tick locations are chosen. The new algorithm
returns a plot view range that is a multiple of the user-requested number of
ticks. This ensures tick marks will be located at whole integers more
consistently. For example, when both y-axes of a``twinx`` plot use
``matplotlib.ticker.LinearLocator`` with the same number of ticks,
their y-tick locations and grid lines will coincide.

`matplotlib.ticker.LogLocator` gains numticks kwarg
---------------------------------------------------

The maximum number of ticks generated by the
`~matplotlib.ticker.LogLocator` can now be controlled explicitly
via setting the new 'numticks' kwarg to an integer.  By default
the kwarg is None which internally sets it to the 'auto' string,
triggering a new algorithm for adjusting the maximum according
to the axis length relative to the ticklabel font size.

`matplotlib.ticker.LogFormatter`: two new kwargs
------------------------------------------------

Previously, minor ticks on log-scaled axes were not labeled by
default.  An algorithm has been added to the
`~matplotlib.ticker.LogFormatter` to control the labeling of
ticks between integer powers of the base.  The algorithm uses
two parameters supplied in a kwarg tuple named 'minor_thresholds'.
See the docstring for further explanation.

To improve support for axes using `~matplotlib.ticker.SymmetricalLogLocator`,
a *linthresh* keyword argument was added.


New defaults for 3D quiver function in mpl_toolkits.mplot3d.axes3d.py
---------------------------------------------------------------------

Matplotlib has both a 2D and a 3D ``quiver`` function. These changes
affect only the 3D function and make the default behavior of the 3D
function match the 2D version. There are two changes:

1) The 3D quiver function previously normalized the arrows to be the
   same length, which makes it unusable for situations where the
   arrows should be different lengths and does not match the behavior
   of the 2D function. This normalization behavior is now controlled
   with the ``normalize`` keyword, which defaults to False.

2) The ``pivot`` keyword now defaults to ``tail`` instead of
   ``tip``. This was done in order to match the default behavior of
   the 2D quiver function.

To obtain the previous behavior with the 3D quiver function, one can
call the function with ::

   ax.quiver(x, y, z, u, v, w, normalize=True, pivot='tip')

where "ax" is an ``Axes3d`` object created with something like ::

   import mpl_toolkits.mplot3d.axes3d
   ax = plt.subplot(111, projection='3d')


Stale figure behavior
---------------------

Attempting to draw the figure will now mark it as not stale (independent if
the draw succeeds).  This change is to prevent repeatedly trying to re-draw a
figure which is raising an error on draw.  The previous behavior would only mark
a figure as not stale after a full re-draw succeeded.


The spectral colormap is now nipy_spectral
------------------------------------------

The colormaps formerly known as ``spectral`` and ``spectral_r`` have been
replaced by ``nipy_spectral`` and ``nipy_spectral_r`` since Matplotlib
1.3.0. Even though the colormap was deprecated in Matplotlib 1.3.0, it never
raised a warning. As of Matplotlib 2.0.0, using the old names raises a
deprecation warning. In the future, using the old names will raise an error.

Default install no longer includes test images
----------------------------------------------

To reduce the size of wheels and source installs, the tests and
baseline images are no longer included by default.

To restore installing the tests and images, use a :file:`setup.cfg` with ::

   [packages]
   tests = True
   toolkits_tests = True

in the source directory at build/install time.
